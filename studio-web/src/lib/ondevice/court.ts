// Court geometry for on-device analysis: the same four-singles-corner calibration as Studio.
// Court metres: x 0–6.1 across from the near-left doubles sideline, y 0–13.4 from the near baseline.

export type Vec2 = [number, number];
export type Matrix3 = number[]; // row-major 3×3

export const COURT_WIDTH = 6.1;
export const COURT_LENGTH = 13.4;
export const NET_Y = COURT_LENGTH / 2;
const SINGLES_CORNERS: Vec2[] = [[0.46, 0], [5.64, 0], [5.64, 13.4], [0.46, 13.4]];

/** Solve the 8-unknown perspective transform mapping four source points onto four destination points. */
export function perspectiveTransform(source: Vec2[], destination: Vec2[]): Matrix3 {
  const rows: number[][] = [];
  for (let index = 0; index < 4; index += 1) {
    const [x, y] = source[index];
    const [u, v] = destination[index];
    rows.push([x, y, 1, 0, 0, 0, -u * x, -u * y, u]);
    rows.push([0, 0, 0, x, y, 1, -v * x, -v * y, v]);
  }
  // Gaussian elimination with partial pivoting on the 8×9 augmented system.
  for (let column = 0; column < 8; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < 8; row += 1) if (Math.abs(rows[row][column]) > Math.abs(rows[pivot][column])) pivot = row;
    if (Math.abs(rows[pivot][column]) < 1e-12) throw new Error("Court corners are degenerate");
    [rows[column], rows[pivot]] = [rows[pivot], rows[column]];
    for (let row = 0; row < 8; row += 1) {
      if (row === column) continue;
      const factor = rows[row][column] / rows[column][column];
      for (let k = column; k < 9; k += 1) rows[row][k] -= factor * rows[column][k];
    }
  }
  const h = rows.map((row, index) => row[8] / row[index]);
  return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
}

export function applyTransform(matrix: Matrix3, [x, y]: Vec2): Vec2 {
  const w = matrix[6] * x + matrix[7] * y + matrix[8];
  return [(matrix[0] * x + matrix[1] * y + matrix[2]) / w, (matrix[3] * x + matrix[4] * y + matrix[5]) / w];
}

export function invert(m: Matrix3): Matrix3 {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const determinant = a * A + b * B + c * C;
  if (Math.abs(determinant) < 1e-12) throw new Error("Court transform is not invertible");
  return [
    A / determinant, -(b * i - c * h) / determinant, (b * f - c * e) / determinant,
    B / determinant, (a * i - c * g) / determinant, -(a * f - c * d) / determinant,
    C / determinant, -(a * h - b * g) / determinant, (a * e - b * d) / determinant,
  ];
}

function cross(o: Vec2, a: Vec2, b: Vec2): number {
  return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
}

/** Corners must be in order (near left, near right, far right, far left) and outline a convex court. */
export function validCorners(corners: Vec2[]): boolean {
  if (corners.length !== 4) return false;
  const turns = corners.map((point, index) => cross(point, corners[(index + 1) % 4], corners[(index + 2) % 4]));
  return turns.every((turn) => turn > 0) || turns.every((turn) => turn < 0);
}

export class Court {
  readonly toCourt: Matrix3;
  readonly toImage: Matrix3;

  /** ``corners`` are normalized image points; ``width``/``height`` the frame size in pixels. */
  constructor(readonly corners: Vec2[], readonly width: number, readonly height: number, readonly doubles: boolean) {
    if (!validCorners(corners)) throw new Error("Tap the four corners in order: near left, near right, far right, far left");
    const pixels = corners.map(([x, y]) => [x * width, y * height] as Vec2);
    this.toCourt = perspectiveTransform(pixels, SINGLES_CORNERS);
    this.toImage = invert(this.toCourt);
  }

  court(pixel: Vec2): Vec2 {
    return applyTransform(this.toCourt, pixel);
  }

  image(court: Vec2): Vec2 {
    return applyTransform(this.toImage, court);
  }

  /** Which half a standing player's feet are on, or null when they are off this court (officials, other courts). */
  sideOf([x, y]: Vec2): "near" | "far" | null {
    const sideline = this.doubles ? 1.2 : 0.7;
    if (x < -sideline || x > COURT_WIDTH + sideline || y < -2.5 || y > COURT_LENGTH + 2.5) return null;
    if (Math.abs(y - NET_Y) < 0.25) return null;
    return y < NET_Y ? "near" : "far";
  }

  /** Image-space crop (pixels) covering one half, with headroom for raised rackets. */
  halfBounds(side: "near" | "far"): { x: number; y: number; w: number; h: number } {
    const [y0, y1] = side === "near" ? [-2.5, NET_Y + 0.4] : [NET_Y - 0.4, COURT_LENGTH + 2.5];
    const points = [[-1.2, y0], [7.3, y0], [7.3, y1], [-1.2, y1]].map((point) => this.image(point as Vec2));
    let x1 = Math.min(...points.map((point) => point[0]));
    let x2 = Math.max(...points.map((point) => point[0]));
    let top = Math.min(...points.map((point) => point[1]));
    let bottom = Math.max(...points.map((point) => point[1]));
    // A standing player's head is ~1.8 m above the feet; give the far half proportionally less headroom.
    top -= (bottom - top) * (side === "near" ? 0.9 : 0.55);
    x1 = Math.max(0, x1); x2 = Math.min(this.width, x2); top = Math.max(0, top); bottom = Math.min(this.height, bottom);
    return { x: x1, y: top, w: Math.max(16, x2 - x1), h: Math.max(16, bottom - top) };
  }

  outline(): Vec2[] {
    const [left, right] = this.doubles ? [0, COURT_WIDTH] : [0.46, 5.64];
    return [[left, 0], [right, 0], [right, COURT_LENGTH], [left, COURT_LENGTH]].map((point) => this.image(point as Vec2));
  }
}
