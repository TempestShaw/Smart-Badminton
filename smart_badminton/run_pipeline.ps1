param(
    [Parameter(Mandatory = $true)][string]$Video,
    [Parameter(Mandatory = $true)][string]$Config,
    [Parameter(Mandatory = $true)][string]$OutputDirectory,
    [string]$Model,
    [string]$CorrectedTimeline,
    [string]$AudioEvents,
    [string]$ShuttleDetections,
    [string]$RenderOutput,
    [string]$PoseModel,
    [string]$Python = 'python',
    [string]$Ffmpeg,
    [string]$Encoder = 'libx264'
)

$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot

function Invoke-PythonStep {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python step failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
$features = Join-Path $OutputDirectory 'smart-features.csv'
$probabilities = Join-Path $OutputDirectory 'rally-probabilities.csv'
$rallies = Join-Path $OutputDirectory 'rallies-auto.csv'
$evaluation = Join-Path $OutputDirectory 'auto-evaluation.json'
$trainingReport = Join-Path $OutputDirectory 'training-report.json'
if (-not $Model) { $Model = Join-Path $workspace 'models\rally-state-final-v4-frozen.joblib' }

if (-not $AudioEvents) {
    $AudioEvents = Join-Path $OutputDirectory 'audio-events.csv'
    $audioArgs = @('-m', 'smart_badminton', 'analyze-audio', '--input', $Video, '--output', $AudioEvents)
    if ($Ffmpeg) { $audioArgs += @('--ffmpeg', $Ffmpeg) }
    Invoke-PythonStep -Arguments $audioArgs
}

$featureArgs = @('-m', 'smart_badminton', 'extract-features', '--video', $Video, '--config', $Config,
    '--output', $features)
if ($PoseModel) { $featureArgs += @('--pose-model', $PoseModel) }
if ($AudioEvents) { $featureArgs += @('--audio-events', $AudioEvents) }
if ($ShuttleDetections) { $featureArgs += @('--shuttle-detections', $ShuttleDetections) }
Invoke-PythonStep -Arguments $featureArgs

if ($CorrectedTimeline) {
    Invoke-PythonStep -Arguments @('-m', 'smart_badminton', 'train', '--features', $features, '--rallies',
        $CorrectedTimeline, '--model', $Model, '--report', $trainingReport)
}
if (-not (Test-Path -LiteralPath $Model)) {
    throw "Model not found. Pass -Model or train one with -CorrectedTimeline: $Model"
}
Invoke-PythonStep -Arguments @('-m', 'smart_badminton', 'predict', '--features', $features, '--model', $Model,
    '--output', $probabilities)
Invoke-PythonStep -Arguments @('-m', 'smart_badminton', 'segment', '--features', $features, '--probabilities',
    $probabilities, '--output', $rallies)
if ($CorrectedTimeline) {
    Invoke-PythonStep -Arguments @('-m', 'smart_badminton', 'evaluate', '--predicted', $rallies, '--truth',
        $CorrectedTimeline, '--output', $evaluation)
}
if ($RenderOutput) {
    $renderArgs = @('-m', 'smart_badminton', 'render', '--video', $Video, '--rallies', $rallies,
        '--output', $RenderOutput, '--encoder', $Encoder)
    if ($Ffmpeg) { $renderArgs += @('--ffmpeg', $Ffmpeg) }
    Invoke-PythonStep -Arguments $renderArgs
}

Write-Host "Automatic timeline: $rallies"
