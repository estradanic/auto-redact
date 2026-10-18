# auto-redact

AI-powered video redaction: redact people from videos based on a runtime-specified prompt.

This tool will:
1. Detect, track, and extract metadata (bounding boxes, keypoints) for people in a video.
2. Find the best snapshot for each tracked person.
3. Use a vision model (VLM or SigLIP) to determine if the person matches a user prompt.
4. Render black boxes over matching people in the output video.

Proof of concept CLI tool. See issues for roadmap.
