import torch
import torch.nn.functional as F
from model import TinyTransformer
from tokenizer import tokenize, detokenize, get_token_to_index
from config import MAX_SEQ_LEN

# --- Assume these are loaded or defined ---
# model = TinyTransformer(...)
# model.load_state_dict(...)
# model.eval()
# ----------------------------------------

def generate_sql(prompt, max_gen_len=50, temperature=0.7):
    """
    Generates SQL using temperature-based sampling and an EOS stopping condition.
    """
    # Get the integer index for special tokens from your vocabulary
    token_to_index = get_token_to_index()
    eos_token_id = token_to_index.get("<EOS>")
    sep_token_id = token_to_index.get("<SEP>")

    # Prepare the initial prompt
    prompt_with_sep = prompt + " <SEP>"
    input_ids = tokenize(prompt_with_sep)
    prompt_len = len(input_ids)
    
    generated = torch.tensor([input_ids])

    for _ in range(max_gen_len):
        # Ensure the input sequence doesn't exceed the model's max length
        input_seq = generated[:, -MAX_SEQ_LEN:]

        with torch.no_grad():
            logits = model(input_seq)
        
        # Get the logits for the very last token
        last_logits = logits[:, -1, :]
        
        # Apply temperature scaling to the logits
        scaled_logits = last_logits / temperature
        
        # Calculate probabilities using softmax
        probs = F.softmax(scaled_logits, dim=-1)
        
        # Sample the next token from the probability distribution
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Stop if the EOS token is generated
        if next_token.item() == eos_token_id:
            break
            
        # Append the new token to our sequence
        generated = torch.cat((generated, next_token), dim=1)

    # Decode the output, skipping the original prompt and <SEP> token
    output_tokens = generated[0][prompt_len:].tolist()
    
    return detokenize(output_tokens)

# Example usage:
# sql_query = generate_sql("Show me all users from Canada")
# print(sql_query)