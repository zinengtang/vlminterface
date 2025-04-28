# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: bigvis
#     language: python
#     name: bigvis
# ---

# %load_ext autoreload
# %autoreload 2
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import json
from PIL import Image
import tyro
import sys
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
print(sys.path)

frames_path = "/home/terran/projects/worldmodel/dreamer_pt/logdir/crafter_reward/train_eps"
outdir = "/home/terran/projects/worldmodel/dreamer_pt/logdir/crafter_reward/train_eps"

N = 500

DENSE_CAPTION_PROMPT = "You are an expert game analyst. Given a single game frame and the list "
"of low‑level key presses executed over the next N frames, output ONE "
"short, high‑level command that best summarises what the agent is trying "
"to do. Reply with an imperative verb phrase, no punctuation."

def main(shard_id: int):

  count = 0

  for example_idx, id_ in tqdm(enumerate(SHARDS[shard_id])):
      if count >= N:
        break

      # Load frames
      file = Path(f"{transcript_path}/transcript_{id_}.json")
      frames_dir = Path(f"{frames_path}/{id_}")
      if not file.exists() or not frames_dir.exists():
          print("Missing transcript or frames", id_)
          continue
      print("Found", id_)
      with open(file) as f:
          transcript = json.load(f)
      frame_paths = [str(fr) for fr in sorted(frames_dir.glob("*jpg"))]
      frames = [Image.open(fr) for fr in frame_paths]

      flat_transcript = [(w["word"], w.get("start")) for seg in transcript["segments"] for w in seg["words"]]
      flat_transcript_clean = []
      last_time = None
      for i, (w, t) in enumerate(flat_transcript):
          if t is None:
              flat_transcript_clean.append((w, last_time))
          else:
              flat_transcript_clean.append((w, t))
              last_time = t
      flat_transcript = flat_transcript_clean
      #print(flat_transcript)
      if len(flat_transcript) == 0:
        print("Skipping", id_)
        continue
      texts, times = zip(*list(flat_transcript))

      text_frames = create_text_frames(texts, times, 1, len(frames))

      example = dict(frame_paths=frame_paths, frames=frames, text_frames=text_frames)

      #display_frames_with_captions(example['frames'][:200],
      #                             example['text_frames'][:200],
      #                             save_path=f"{outdir}/subtitles/{id_}.jpg",
      #                             num_cols=5)

      # Dense caption
      def format_message(text_frames, start=0, step=None):
          timed_subtitles = []
          if step is None:
              step = 10000
          for i, fr in enumerate(text_frames):
              if i < start: continue
              if i >= start+step: break
              if len(fr) == 0: continue
              timed_subtitles.append(f"{i} {fr}")

          return msg_template.format(subtitles="\n".join(timed_subtitles))

      def parse_densecap_output_into_frames(out, nframes):
        events = {}
        for line in out.strip().split("\n"):
            if line.startswith("```"): continue
            if len(line.strip()) == 0: continue
            timestamp = line.strip().split(" ", 1)[0]
            if "-" in timestamp:  # Model outputs range, e.g. 73-79
              timestamp = timestamp.split("-")[0]
            if not timestamp.isdigit():
                print("BAD:", line)
                continue
            print(line)
            time, event = line.strip().split(" ", 1)
            events[time] = event
        text_frames = []
        for i in range(nframes):
            if str(i) in events:
                text_frames.append(events[str(i)])
            else:
                text_frames.append("")
        return text_frames

      dense_captioner = GPT4(DENSE_CAPTION_PROMPT, model="gpt-4o")
      #print("Encoding", frame_paths)
      encoded_imgs = [dense_captioner.encode_image(fr) for fr in frame_paths]
      # Annotate clip-by-clip
      clip_outputs = ""
      step = 30
      for start in range(0, len(encoded_imgs), step):
          message_content = [{"type": "text", "text": "Video (the text before each frame is the `second` timestamp for that frame."}]
          for i, base64_image in enumerate(encoded_imgs):
              if i < start: continue
              if i >= start + step: break
              message_content.append({"type": "text", "text": f"Time (s): {i}"})
              message_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
          messages = [
              {"role": "user", "content": message_content},
              {"role": "user", "content": format_message(example['text_frames'], start, step)}
          ]
          #print(messages[0]["content"][0])
          #print(messages[0]["content"][1])
          #print(messages[-1])
          for _ in range(3):
            out = dense_captioner.respond_raw(messages=messages)
            if "I'm sorry" not in out:
              break

          clip_outputs += out
          print(out)

      example["generated_captions"] = parse_densecap_output_into_frames(clip_outputs, len(frames))

      with open(f"{outdir}/densecap/{id_}.json", "w") as f:
          json.dump(example["generated_captions"], f)

      with open(f"{outdir}/densecap/{id_}.txt", "w") as f:
        for fr in example["generated_captions"]:
          f.write(fr + "\n")

      if example_idx % 100 == 0:
        display_frames_with_captions(
          frames[:100], example["generated_captions"][:100], num_cols=3,
          save_path=f"{outdir}/densecap/{id_}.jpg")

      def format_message_dialogue(text_frames, start=0, step=None):
          timed_subtitles = []
          if step is None:
              step = 10000
          for i, fr in enumerate(text_frames):
              if i < start: continue
              if i >= start+step: break
              if len(fr) == 0: continue
              timed_subtitles.append(f"{fr}")  # Timestamp included already

          return msg_template_dialogue.format(subtitles_and_captions="\n".join(timed_subtitles))

      def parse_partial_dialogue(out, start, step):
          events = defaultdict(str)
          for line in out.strip().split("\n"):
              if line.startswith("```"): continue
              if len(line.strip()) == 0: continue
              if not line.strip().split(" ", 1)[0].isdigit():
                  print("BAD:", line)
                  continue
              print(line)
              if len(line.strip().split(" ", 1)) != 2:
                  print("BAD:", line)
                  continue
              time, event = line.strip().split(" ", 1)
              events[time] += "\n" + event
          text_frames = []
          for i in range(start, start+step):
              if str(i) in events:
                  text_frames.append(events[str(i)].strip())
              else:
                  text_frames.append("")
          return text_frames

      dialogue_generator = GPT4(DIALOGUE_PROMPT, model="gpt-4o")

      dialogue_clip_outputs = ""
      step = 15
      subtitles_and_caps = [
          f'{i} "{sub}" ({cap})' for i, (sub, cap) in enumerate(zip(
              example['text_frames'], example['generated_captions']
          ))
      ]
      dialogue_parsed = []
      for start in range(0, len(encoded_imgs), step):
          ctx = format_message_dialogue(subtitles_and_caps, start, step)
          print(ctx)
          print("---"*20)
          for _ in range(3):
            out = dialogue_generator.respond(ctx, stop=["\n\n", "Video Subtitles"])
            if out is not None:
              dialogue_clip_outputs += "\n" + out
              break
          dialogue_parsed.extend(parse_partial_dialogue(out, start, min(step, len(subtitles_and_caps[start: start+step]))))
          print(out)
          print("==="*20)

      example["generated_dialogue_raw"] = dialogue_clip_outputs
      example["generated_dialogue_frames"] = dialogue_parsed

      with open(f"{outdir}/dialogue/{id_}.json", "w") as f:
        json.dump({"raw": example["generated_dialogue_raw"], "frames":
                   example["generated_dialogue_frames"]}, f)

      with open(f"{outdir}/dialogue/{id_}.txt", "w") as f:
        for fr in example["generated_dialogue_frames"]:
          f.write(fr + "\n")

      if example_idx % 100 == 0:
        display_frames_with_captions(
          frames[:200], example["generated_dialogue_frames"][:200], num_cols=3,
          save_path=f"{outdir}/dialogue/{id_}.jpg")

      count += 1

if __name__ == "__main__":
  tyro.cli(main)
