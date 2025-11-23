
import os 
import torch

import torch.nn as nn 

from transformers import ViTModel, GPT2LMHeadModel, GPT2Config
from transformers import GPT2Tokenizer
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# GPT-2 doesn’t have a pad token, so set one
tokenizer.pad_token = tokenizer.eos_token
class ViTGPT2Captioner(nn.Module):
    def __init__(self, vit_name="google/vit-base-patch16-224", gpt2_name="gpt2",  tokenizer=tokenizer):
        super().__init__()
        # Encoder
        self.vit = ViTModel.from_pretrained(vit_name)
        
        # Decoder with cross-attention
        gpt2_config = GPT2Config.from_pretrained(gpt2_name)
        gpt2_config.add_cross_attention = True
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_name, config=gpt2_config)
        # Optional: projection if dims mismatch
        if self.vit.config.hidden_size != gpt2_config.hidden_size:
            print("The hidden size of both models is different")
            print(f"Vit Hidden size = {self.vit.config.hidden_size}, GPT2 hidden Size = {gpt2_config.hidden_size}")
            self.proj = nn.Linear(self.vit.config.hidden_size, gpt2_config.hidden_size)
        else:
            print("Hidden size are same")
            self.proj = None
        self.tokenizer = tokenizer

    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):

        # 1. Encode image
        vit_outputs = self.vit(pixel_values=pixel_values)
        encoder_hidden_states = vit_outputs.last_hidden_state  # [B, seq, dim]
        if self.proj:
            encoder_hidden_states = self.proj(encoder_hidden_states)
        # 2. Decode text
        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            labels=labels
        )
        print(f"Outputs from the GPT2 model (): ", outputs)
        return outputs
    # def generate(self, image, max_length=50, top_p=0.9, device="cuda"):
    #     self.eval()
    #     print("Inside forward of Normal Model, printing initial input:")
    #     with torch.no_grad():
    #         # Encode image
    #         image = image.unsqueeze(0).to(device)  # [1, C, H, W]
    #         vit_outputs = self.vit(pixel_values=image)
    #         encoder_hidden_states = vit_outputs.last_hidden_state
    #         if self.proj:
    #             encoder_hidden_states = self.proj(encoder_hidden_states)

    #         # Start caption with <BOS> (use EOS if tokenizer has no BOS)
    #         caption = [self.tokenizer.bos_token_id if self.tokenizer.bos_token_id else self.tokenizer.eos_token_id]

    #         for _ in range(max_length):
    #             input_ids = torch.tensor(caption, dtype=torch.long).unsqueeze(0).to(device)

    #             outputs = self.gpt2(
    #                 input_ids=input_ids,
    #                 encoder_hidden_states=encoder_hidden_states
    #             )
    #             logits = outputs.logits  # [1, seq_len, vocab]
    #             last_logits = logits[0, -1, :]

    #             # Apply nucleus (top-p) sampling
    #             probs = torch.softmax(last_logits, dim=-1)
    #             sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    #             cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    #             # mask out tokens past cumulative prob threshold
    #             sorted_indices_to_remove = cumulative_probs > top_p
    #             sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    #             sorted_indices_to_remove[..., 0] = 0

    #             indices_to_remove = sorted_indices[sorted_indices_to_remove]
    #             probs[indices_to_remove] = 0.0
    #             probs = probs / torch.sum(probs)

    #             next_token = torch.multinomial(probs, num_samples=1).item()
    #             caption.append(next_token)

    #             if next_token == self.tokenizer.eos_token_id:
    #                 break

    #     return self.tokenizer.decode(caption, skip_special_tokens=True)

    def generate(self, image, max_length=50, top_p=0.9, device="cuda"):
        """
        Generates a caption for an image autoregressively with detailed logging.
        """
        self.eval()
        print("=========================================================")
        print("====== Starting Autoregressive Caption Generation ======")
        print(f"       max_length={max_length}, top_p={top_p}, device='{device}'")
        print("=========================================================\n")

        with torch.no_grad():
            # --- 1. ENCODER STEP: Process the image ---
            image = image.unsqueeze(0).to(device)
            print(f"Input image tensor shape: {image.shape}")

            # Get image features from the ViT encoder
            vit_outputs = self.vit(pixel_values=image)
            encoder_hidden_states = vit_outputs.last_hidden_state
            if self.proj:
                encoder_hidden_states = self.proj(encoder_hidden_states)

            print(f"Encoder output shape (context): {encoder_hidden_states.shape}")
            print("Encoder output slice [0, :3, :5]:\n", encoder_hidden_states[0, :3, :5])

            # --- 2. DECODER SETUP: Initialize the caption ---
            # Start with the Beginning-Of-Sequence token
            start_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.eos_token_id
            caption = [start_token_id]
            print(f"\nStarting caption with token ID: {start_token_id} ('{self.tokenizer.decode(start_token_id)}')\n")

            # --- 3. AUTOREGRESSIVE GENERATION LOOP ---
            for i in range(max_length):
                print(f"----------------- Step {i+1} -----------------")
                input_ids = torch.tensor(caption, dtype=torch.long).unsqueeze(0).to(device)
                print(f"Decoder Input IDs: {input_ids.tolist()}")
                print(f"Decoded Input:     '{self.tokenizer.decode(caption)}'")

                # --- Forward pass through the GPT-2 decoder ---
                outputs = self.gpt2(
                    input_ids=input_ids,
                    encoder_hidden_states=encoder_hidden_states
                )
                logits = outputs.logits  # Shape: [1, seq_len, vocab_size]
                
                # Select logits for the last token only, as that's our prediction
                last_logits = logits[0, -1, :] # Shape: [vocab_size]
                print(f"Logits shape for last token: {last_logits.shape}")

                # Print top 5 predicted logits for debugging what the model is "thinking"
                top_k_logits, top_k_indices = torch.topk(last_logits, 5)
                print("Top 5 predicted logits:")
                for k in range(5):
                    token_str = self.tokenizer.decode(top_k_indices[k].item())
                    print(f"  - Token '{token_str}' (ID: {top_k_indices[k].item()}): {top_k_logits[k].item():.4f}")

                # --- Nucleus (top-p) Sampling ---
                probs = torch.softmax(last_logits, dim=-1)
                
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Create a mask for tokens to remove based on cumulative probability
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the mask to ensure we keep at least one token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                
                # Apply the mask by setting low-probability tokens to 0
                probs[indices_to_remove] = 0.0
                
                # Renormalize the probabilities
                probs = probs / torch.sum(probs)

                print(f"Sampling from {torch.count_nonzero(probs)} tokens after top-p (p={top_p}) filtering.")

                # Sample the next token from the filtered distribution
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                # Append the new token to our caption sequence
                caption.append(next_token)
                
                next_token_str = self.tokenizer.decode(next_token)
                print(f"==> Selected next token: {next_token} ('{next_token_str}')\n")

                # Check for the end-of-sequence token
                if next_token == self.tokenizer.eos_token_id:
                    print("End-of-sequence token generated. Stopping.")
                    break
        
        # --- 4. FINAL DECODING ---
        print("=========================================================")
        print("============== Generation Finished ==============")
        print(f"Final token IDs: {caption}")
        final_caption = self.tokenizer.decode(caption, skip_special_tokens=True)
        print(f"Final Decoded Caption: '{final_caption}'")
        print("=========================================================\n")

        return final_caption

    