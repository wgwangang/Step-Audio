import argparse
import os
import torchaudio
from stepaudio import StepAudio


def main():
    parser = argparse.ArgumentParser(description="StepAudio Offline Inference")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Base path for model files"
    )
    parser.add_argument(
        "--output-path", type=str, required=True, help="Base path for response audio files"
    )
    args = parser.parse_args()

    model = StepAudio(
        tokenizer_path=f"{args.model_path}/Step-Audio-Tokenizer",
        tts_path=f"{args.model_path}/Step-Audio-TTS-3B",
        llm_path=f"{args.model_path}/Step-Audio-Chat",
    )

    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # example for text input
    input_text = "你好，我是你的朋友，我叫小明，你叫什么名字？"
    text, audio, sr = model(
        [{"role": "user", "content": input_text}],
        "Tingting",
    )
    print(f"input text: {input_text}")
    print(f"response text: {text}")
    response_audio_path1 = f"{output_path}/output_e2e_tqta1.wav"
    torchaudio.save(response_audio_path1, audio, sr)
    # torchaudio.save("output/output_e2e_tqta.wav", audio, sr)

    # example for audio input
    text, audio, sr = model(
        [
            {
                "role": "user",
                # "content": {"type": "audio", "audio": "output/output_e2e_tqta.wav"},
                "content": {"type": "audio", "audio": response_audio_path1},
            }
        ],
        "Tingting",
    )
    print(f"response text: {text}")
    response_audio_path2 = f"{output_path}/output_e2e_aqta2.wav"
    torchaudio.save(response_audio_path2, audio, sr)
    # torchaudio.save("output/output_e2e_aqta.wav", audio, sr)


if __name__ == "__main__":
    main()
