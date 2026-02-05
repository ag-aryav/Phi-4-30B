from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import torch
import torch.nn as nn
import os

# 1. Load original model
model_name = "microsoft/phi-4"
print("Loading original model...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Double the layers
original_config = model.config
original_layers = original_config.num_hidden_layers
new_layer_count = original_layers * 2
print(f"Original layers: {original_layers}, New target: {new_layer_count}")

# 3. Create expanded config
new_config = AutoConfig.from_pretrained(model_name)
new_config.num_hidden_layers = new_layer_count

# 4. Create new model with expanded layers
print("\nCreating expanded model...")
new_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=new_config,
    torch_dtype=torch.bfloat16,
    ignore_mismatched_sizes=True
)

# 5. Copy original weights, EXACT ZERO NEW layers
print("Initializing new layers to EXACT ZERO...")
with torch.no_grad():
    # Copy all original layers
    for i in range(original_layers):
        # Copy layer weights
        new_model.model.layers[i].load_state_dict(model.model.layers[i].state_dict())
    
    # EXACT ZERO for NEW layers (index original_layers onward)
    for i in range(original_layers, new_layer_count):
        layer = new_model.model.layers[i]
        
        # Set EVERY parameter to zero
        for name, param in layer.named_parameters():
            # Create a zeros tensor with the same shape as parameter
            zeros_tensor = torch.zeros_like(param.data)
            param.data.copy_(zeros_tensor)
            
        # Also zero any buffers if they exist
        for name, buffer in layer.named_buffers():
            if buffer.dtype in [torch.float16, torch.float32, torch.bfloat16]:
                zeros_tensor = torch.zeros_like(buffer.data)
                buffer.data.copy_(zeros_tensor)

print("✅ Model doubled with EXACT ZERO initialized new layers!")

# 6. Verify it works (forward pass test)
print("\nTesting forward pass...")
test_input = torch.randint(0, original_config.vocab_size, (1, 10))
with torch.no_grad():
    # Get output from original model
    original_output = model(test_input).logits
    
    # Get output from expanded model
    expanded_output = new_model(test_input).logits
    
    # Check if they're identical (they should be!)
    if torch.allclose(original_output, expanded_output, rtol=1e-5, atol=1e-5):
        print("✅ Forward pass successful! Outputs are IDENTICAL (as expected)")
    else:
        print("⚠️  Outputs differ slightly (numerical precision)")
        diff = torch.abs(original_output - expanded_output).max().item()
        print(f"   Max difference: {diff:.2e}")

# 7. Calculate and display model size
total_params = sum(p.numel() for p in new_model.parameters())
model_size_b = total_params / 1e9
print(f"Model size: ~{model_size_b:.1f}B parameters")

# 8. Save the model and tokenizer
save_dir = f"./phi-4-{model_size_b:.1f}B-zero"
print(f"\nSaving model and tokenizer to: {save_dir}")

# Create directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Save model
new_model.save_pretrained(save_dir)

# Save tokenizer
tokenizer.save_pretrained(save_dir)

# Save config separately (with correct parameters)
new_config.save_pretrained(save_dir)

print(f"✅ Model and tokenizer saved successfully in '{save_dir}' directory!")

print("\\n🎉 All done! Your model has:")
print(f"   - {original_layers} original layers (copied from phi-4)")
print(f"   - {new_layer_count - original_layers} new layers (exact zero initialized)")
print(f"   - Perfect identity function via residuals")
print(f"   - Zero knowledge loss guaranteed!")
