import wave
import os

input_file = "audio2.wav"
output_folder = "dataset/location_B"

os.makedirs(output_folder, exist_ok=True)

segment_seconds = 10

with wave.open(input_file, "rb") as audio:

    frame_rate = audio.getframerate()
    frames_per_segment = int(frame_rate * segment_seconds)

    total_frames = audio.getnframes()
    segment_number = 1

    for start in range(0, total_frames, frames_per_segment):

        audio.setpos(start)

        frames = audio.readframes(frames_per_segment)

        output_file = os.path.join(
            output_folder,
            f"clip2_{segment_number:03d}.wav"
        )

        with wave.open(output_file, "wb") as output:

            output.setnchannels(audio.getnchannels())
            output.setsampwidth(audio.getsampwidth())
            output.setframerate(audio.getframerate())

            output.writeframes(frames)

        print(f"Created: {output_file}")

        segment_number += 1