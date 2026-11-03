# Live analysis

Point a camera at the court, play, and watch the rally clock, shot count, player movement and court positions
update on any phone, tablet or laptop on the same Wi-Fi. A computer does the analysis; viewers only need a browser.

No computer at the court? [Phone only](#phone-only-no-computer) runs everything on the phone itself.

Live analysis gives quick feedback while you play. For the editable highlight cut, record the session and run the full
offline analysis in Studio afterwards. The offline pipeline sees every frame, uses the match sound and the whole
video's context, and stays the accurate result.

## Start a live session

Install the vision stack and models once (see the main README), then start the live server on the computer that runs
the analysis:

```bash
smart-badminton live --library /path/to/matches --host 0.0.0.0
```

It prints the address to open, for example `http://192.168.1.20:8765`. Open it on the computer, or on a phone on the
same network to watch from the court. In a normal Studio session, the **Live** button in the top bar opens the same
page. Studio needs `--host 0.0.0.0` before other devices can reach it.

Choose a video source:

| Source | When to use it | What to enter |
| --- | --- | --- |
| This computer's camera | A webcam or USB camera on the analysis computer | Camera number, usually `0` |
| This device's camera | The phone is the camera; the computer analyses | Nothing; needs HTTPS (below) |
| Stream or file | An IP-camera app, an RTSP camera, or a recording to replay as if live | `rtsp://…`, `http://phone-ip:8080/video`, or a file path |

Choose **Singles** or **Doubles**. In doubles, Studio follows two players on each side.

### Using a phone as the camera

Phone browsers only share their camera with HTTPS pages. Start the server with a local certificate:

```bash
smart-badminton live --library /path/to/matches --host 0.0.0.0 --https
```

Open the printed `https://` address on the phone. The browser warns once about the self-signed certificate; accept it
for this address. Pick **This device's camera** and start. The phone streams about eight frames per second and its
microphone to the computer. If you also tick **Record**, the phone keeps a full-quality recording with sound and
uploads it to the library's `Live/` folder when you press **End**. Open it in Studio for the full analysis.

An IP-camera app (for example one that serves `http://phone-ip:8080/video`) also works without HTTPS. Enter its
address under **Stream or file**.

## Calibrate the court once

The first time a camera position is used, Studio shows a still frame. Tap the four corners where the **singles**
sidelines meet the baselines, in this order: near left, near right, far right, far left. From those four points
Studio derives the court outline (singles or doubles), both halves, the net and the shuttle airspace, and from then on
measures positions in metres. Recalibrate whenever the camera moves.

If Studio already has a calibration for the same camera position, **Use Studio's calibration** copies it.

**Corners out of frame.** A camera beside the court or low behind it often cannot see all four corners. Choose
**Corners out of frame** and mark any four or more court points you can see clearly instead: pick a point on the small
court plan (net-post feet, the ends and centre of each short service line, the baseline corners and centres), then tap
it in the picture. The points must not all lie on one line. Studio fits the court to them and works out where the
corners would be, even outside the picture. It rejects points that disagree with each other, which usually means a
point was given the wrong name.

Mount the camera high behind one baseline with the whole court in view. Avoid wide-angle distortion where possible.

## What live analysis measures

- **Rally state and duration.** Rallies start and end from shuttle flight and, when sound is available, the rally
  model's racket-hit evidence. The model's own answer needs two seconds of context, so it lags slightly; shuttle
  flight starts the clock sooner.
- **Shots per rally and hits per player.** A shot is counted when the shuttle changes sides or a swinging wrist meets
  it. At live frame rates this is an estimate; the offline contact analysis is more precise.
- **Player tracking.** Each player keeps a stable ID (`Near 1`, `Near 2`, `Far 1`, `Far 2`), even when doubles partners
  cross or one hides the other. Movement distance, current and top speed come from calibrated court metres.
- **Swings.** Fast wrist movements per player.

## Phone only (no computer)

The on-device mode runs pose tracking, hit sounds and rally detection inside the phone's browser. Nothing is uploaded,
and no computer or Wi-Fi is needed once the page has loaded. Open the hosted Studio page (the web build in
`studio-web`, for example on Vercel), tap **Live** in the top bar, or add `?engine=device` to the address. Choose
singles or doubles, allow the camera and microphone, then calibrate as above. The calibration is remembered on this
phone.

What it measures: each player's ID, court position, distance, speed and swings; hits, where a racket sound must match
a swing on this court within 0.2 s (sounds from neighbouring courts or talking don't count); and rallies, which open
on the serve exchange and close after 2.6 s without a hit. It does not track the shuttle, so shot counts depend on
hearing the hits: in a noisy hall, or with the microphone blocked, only strong swings are counted.

Speed comes from running the pose model only where the players are. Each tracked player gets a 192-pixel crop
instead of the whole frame, and a half-court is only scanned again when a player is lost, or every 3 seconds. The
phone uses its GPU through WebGPU when it has one (Safari on iOS 26 and later, recent Chrome) and is faster that
way; otherwise it uses all but one CPU core with an 8-bit version of the model (3.4 MB instead of 12 MB, with about the
same wrist accuracy). It times both at start-up and keeps the faster one. On a four-core laptop CPU without a GPU, in
a browser, this gives about 17 analysed frames per second. The badges under the picture show the backend, frame rate
and inference time on your phone.

Keep the phone plugged in or on a power bank for long sessions: continuous camera and inference drain the battery.

**Hosting the page.** The page needs HTTPS (for the camera) and these two headers (for multi-threaded inference):
`Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`. `studio-web/vercel.json`
sets them. The two ONNX models are built from the verified pose checkpoint:

```bash
smart-badminton install-models --directory models
pip install "smart-badminton[ondevice]"
smart-badminton export-ondevice-models --directory models
```

`npm run build` in `studio-web` copies them into the site. A build without local models (Vercel) downloads them from
`ONDEVICE_MODELS_URL`, or by default from this repository's latest GitHub release, which publishes
`yolo11n-pose.onnx` and `yolo11n-pose-int8.onnx`. The phone caches the model after the first visit. The models are
AGPL-3.0, like the checkpoint they come from; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Speed and accuracy

On a four-core laptop CPU, live analysis runs at about 6–7 frames per second with pose and shuttle tracking. An
NVIDIA GPU runs both models in half precision and reaches the camera's frame rate. To trade accuracy for speed on a
CPU:

- lower the inference sizes: `--pose-size 384 --shuttle-size 512` (pose at 480 instead of 640 already cuts its time
  by about 40%; smaller sizes lose far-court detail);
- untick **Track the shuttle**, or start with `--no-shuttle` (rally detection then relies on the rally model);
- on Intel CPUs, an OpenVINO export of the pose model may be faster:
  `yolo export model=models/yolo11n-pose.pt format=openvino`, then pass the exported folder with `--pose-model`.
  Compare the frame-rate readout; an ONNX export was not faster than PyTorch in our tests.

The live page shows the analysed frame rate. Frames that arrive while the computer is busy are dropped, never queued,
so the view stays current.
