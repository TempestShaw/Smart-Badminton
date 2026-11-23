// Calibrating from court landmarks when some singles corners are out of frame (a camera beside the court, or low
// behind it). Any four or more known painted points give a least-squares homography; projecting the singles
// corners through it yields "virtual" corners (possibly outside the frame), so everything corner-based still works.
// Ids and positions match smart_badminton/geometry.py COURT_LANDMARKS_METERS.

export type Point = [number, number];

export interface CourtLandmark {
  id: string;
  court: Point; // metres: x across from the near-left doubles sideline, y from the near baseline
  label: string; // i18n key
}

export const COURT_LANDMARKS: CourtLandmark[] = [
  { id: "near-left-corner", court: [0.46, 0], label: "近左角" },
  { id: "near-centre-baseline", court: [3.05, 0], label: "近底线中点" },
  { id: "near-right-corner", court: [5.64, 0], label: "近右角" },
  { id: "near-left-service", court: [0.46, 4.72], label: "近端前发球线左端" },
  { id: "near-centre-service", court: [3.05, 4.72], label: "近端前发球线中点" },
  { id: "near-right-service", court: [5.64, 4.72], label: "近端前发球线右端" },
  { id: "left-post", court: [0, 6.7], label: "左网柱底" },
  { id: "right-post", court: [6.1, 6.7], label: "右网柱底" },
  { id: "far-left-service", court: [0.46, 8.68], label: "远端前发球线左端" },
  { id: "far-centre-service", court: [3.05, 8.68], label: "远端前发球线中点" },
  { id: "far-right-service", court: [5.64, 8.68], label: "远端前发球线右端" },
  { id: "far-left-corner", court: [0.46, 13.4], label: "远左角" },
  { id: "far-centre-baseline", court: [3.05, 13.4], label: "远底线中点" },
  { id: "far-right-corner", court: [5.64, 13.4], label: "远右角" },
];

const SINGLES_CORNERS: Point[] = [[0.46, 0], [5.64, 0], [5.64, 13.4], [0.46, 13.4]];

export class LandmarkError extends Error {}

function solve(matrix: number[][], vector: number[]): number[] {
  const size = vector.length;
  const rows = matrix.map((row, index) => [...row, vector[index]]);
  for (let column = 0; column < size; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < size; row += 1) if (Math.abs(rows[row][column]) > Math.abs(rows[pivot][column])) pivot = row;
    if (Math.abs(rows[pivot][column]) < 1e-12) throw new LandmarkError("degenerate");
    [rows[column], rows[pivot]] = [rows[pivot], rows[column]];
    for (let row = 0; row < size; row += 1) {
      if (row === column) continue;
      const factor = rows[row][column] / rows[column][column];
      for (let k = column; k <= size; k += 1) rows[row][k] -= factor * rows[column][k];
    }
  }
  return rows.map((row, index) => row[size] / row[index]);
}

/** Similarity that centres points and scales their mean distance to √2 (Hartley normalisation). */
function normaliser(points: Point[]): { apply: (point: Point) => Point; matrix: number[] } {
  const cx = points.reduce((sum, point) => sum + point[0], 0) / points.length;
  const cy = points.reduce((sum, point) => sum + point[1], 0) / points.length;
  const mean = points.reduce((sum, point) => sum + Math.hypot(point[0] - cx, point[1] - cy), 0) / points.length || 1;
  const scale = Math.SQRT2 / mean;
  return { apply: ([x, y]) => [(x - cx) * scale, (y - cy) * scale], matrix: [scale, 0, -cx * scale, 0, scale, -cy * scale, 0, 0, 1] };
}

function multiply(a: number[], b: number[]): number[] {
  const out = new Array(9).fill(0);
  for (let row = 0; row < 3; row += 1) for (let column = 0; column < 3; column += 1) {
    for (let k = 0; k < 3; k += 1) out[row * 3 + column] += a[row * 3 + k] * b[k * 3 + column];
  }
  return out;
}

function inverseSimilarity(m: number[]): number[] {
  const scale = m[0];
  return [1 / scale, 0, -m[2] / scale, 0, 1 / scale, -m[5] / scale, 0, 0, 1];
}

function project(h: number[], [x, y]: Point): { point: Point; w: number } {
  const w = h[6] * x + h[7] * y + h[8];
  return { point: [(h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w], w };
}

/** Least-squares homography from court metres to normalised image points (h33 = 1, normalised DLT). */
export function landmarkHomography(pairs: { court: Point; image: Point }[]): number[] {
  const source = normaliser(pairs.map((pair) => pair.court));
  const target = normaliser(pairs.map((pair) => pair.image));
  const normal = Array.from({ length: 8 }, () => new Array(8).fill(0));
  const right = new Array(8).fill(0);
  for (const pair of pairs) {
    const [x, y] = source.apply(pair.court);
    const [u, v] = target.apply(pair.image);
    for (const [row, value] of [[[x, y, 1, 0, 0, 0, -u * x, -u * y], u], [[0, 0, 0, x, y, 1, -v * x, -v * y], v]] as [number[], number][]) {
      for (let i = 0; i < 8; i += 1) {
        right[i] += row[i] * value;
        for (let j = 0; j < 8; j += 1) normal[i][j] += row[i] * row[j];
      }
    }
  }
  const h = [...solve(normal, right), 1];
  return multiply(multiply(inverseSimilarity(target.matrix), h), source.matrix);
}

/**
 * Singles corners (near left → near right → far right → far left, normalised image coordinates) implied by four or
 * more placed landmarks. Throws a LandmarkError whose message is an i18n key.
 */
export function cornersFromLandmarks(placed: { id: string; image: Point }[]): Point[] {
  const pairs = placed.map((item) => {
    const landmark = COURT_LANDMARKS.find((candidate) => candidate.id === item.id);
    if (!landmark) throw new LandmarkError("未知的场地点");
    return { court: landmark.court, image: item.image };
  });
  if (pairs.length < 4) throw new LandmarkError("至少标出 4 个场地点");
  // The points must span two directions on the court, not lie along one line.
  const cx = pairs.reduce((sum, pair) => sum + pair.court[0], 0) / pairs.length;
  const cy = pairs.reduce((sum, pair) => sum + pair.court[1], 0) / pairs.length;
  let sxx = 0, syy = 0, sxy = 0;
  for (const { court: [x, y] } of pairs) { sxx += (x - cx) ** 2; syy += (y - cy) ** 2; sxy += (x - cx) * (y - cy); }
  const smallest = (sxx + syy) / 2 - Math.sqrt(((sxx - syy) / 2) ** 2 + sxy ** 2);
  if (Math.sqrt(Math.max(0, smallest)) < 0.5) throw new LandmarkError("这些点在同一条线上，请再标一个别的线上的点");
  let h: number[];
  try {
    h = landmarkHomography(pairs);
  } catch {
    throw new LandmarkError("这些点无法组成一致的场地视角");
  }
  const error = Math.max(...pairs.map((pair) => {
    const { point } = project(h, pair.court);
    return Math.hypot(point[0] - pair.image[0], point[1] - pair.image[1]);
  }));
  if (!Number.isFinite(error) || error > 0.04) throw new LandmarkError("有的点和其他点对不上，请检查点的名称和位置");
  const corners = SINGLES_CORNERS.map((corner) => project(h, corner));
  if (!(corners.every((corner) => corner.w > 0) || corners.every((corner) => corner.w < 0))) {
    throw new LandmarkError("有一部分场地在镜头后面，请把手机往后放");
  }
  return corners.map(({ point }) => [Number(point[0].toFixed(5)), Number(point[1].toFixed(5))]);
}
