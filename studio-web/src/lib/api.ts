const configuredOrigin = process.env.NEXT_PUBLIC_STUDIO_API_ORIGIN?.replace(/\/$/, "") ?? "";

export function apiUrl(path: string): string {
  return `${configuredOrigin}${path}`;
}

export async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(apiUrl(path), {
    ...init,
    headers: init?.body ? { "Content-Type": "application/json", ...init.headers } : init?.headers,
  });
  const payload = (await response.json()) as T & { detail?: string };
  if (!response.ok) throw new Error(payload.detail || `Request failed: ${response.status}`);
  return payload;
}

export function mediaUrl(path: string): string {
  return apiUrl(path);
}
