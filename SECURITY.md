# Security policy

Smart Badminton is a loopback-only local media tool. Do not expose the Studio port to an untrusted network: the local
filesystem browser and media endpoints are intentionally designed for the person running the process on that machine.

Please report a vulnerability through the repository host's private security-advisory feature. Include the affected
version, operating system, reproduction steps and whether the service was still bound to `127.0.0.1`. Do not attach
private match footage or model weights; use the privacy-safe metadata fixture when possible.

Security fixes target the latest release. A report is out of scope when it requires an already-compromised local user
account or deliberately starting Studio on a public interface despite the documented loopback requirement.
