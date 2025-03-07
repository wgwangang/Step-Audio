import argparse
import os
import time
import torch
import torchaudio

from stepaudio4training import StepAudio

from transformers import get_linear_schedule_with_warmup

from peft import LoraConfig, TaskType, get_peft_model

input_ids = [[   1,     4,  28504,    78,  1027,  922,   325, 22227,  1027, 20623,
           325,   676,  2120,   503,   684,   325,  1027,  2120,  3559,  9772,
          1216,     3,     4, 56446,    78,   1027,  922,   325,   503,   684,
           325,   676,  2120,   503,  2572, 330]]

input_ids = torch.tensor(input_ids)

batch = {
    "input_ids": input_ids,
    "attention_mask": torch.ones_like(input_ids),
    "labels": input_ids,
    }

def main():
    parser = argparse.ArgumentParser(description="StepAudio Offline Inference")
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Base path for model files"
    )
    parser.add_argument(
        "--output-path", type=str, required=True,
        help="Base path for response audio files"
    )
    args, _ = parser.parse_known_args()

    model = StepAudio(
        tokenizer_path=f"{args.model_path}/Step-Audio-Tokenizer",
        tts_path=f"{args.model_path}/Step-Audio-TTS-3B",
        llm_path=f"{args.model_path}/Step-Audio-Chat",
    )

    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # example for text input
    i = 0
    while (True):
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

        if i == 0:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                target_modules=["q_proj",
                                "k_proj",
                                "v_proj",
                                ],
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,)
            model1 = get_peft_model(model.llm, peft_config)
            model1.print_trainable_parameters()

            # optimizer
            optimizer = torch.optim.AdamW(model1.parameters(), lr=0.0001)

            num_epochs = 20
            step_num_per_epoch = 100

            # lr scheduler
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=step_num_per_epoch * num_epochs,
            )

            device = "cuda"
            model1 = model1.to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            global_step = 0
            for _ in range(num_epochs * step_num_per_epoch):
                model1.train()
                outputs = model1(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                # Update the importance of low-rank matrices
                # and allocate the budget accordingly.
                model1.base_model.update_and_allocate(global_step)
                optimizer.zero_grad()
                global_step += 1
                print(loss)

        time.sleep(3000)
        i += 1

if __name__ == "__main__":
    main()
