from transformers import pipeline
import pandas as pd

# Load GPT-2 text generation model
generator = pipeline(
    "text-generation",
    model="gpt2"
)

# Read prompts from prompts.txt
with open("dataset/prompts.txt", "r", encoding="utf-8") as file:
    prompts = [line.strip() for line in file if line.strip()]

# Store results
results = []

# Values to test
temperatures = [0.2, 0.7, 1.0]
top_k_values = [20, 50, 100]

# Generate text for each combination
for prompt in prompts:

    for temperature in temperatures:

        for top_k in top_k_values:

            output = generator(
                prompt,
                max_new_tokens=50,
                temperature=temperature,
                top_k=top_k,
                do_sample=True,
                num_return_sequences=1
            )

            generated_text = output[0]["generated_text"]

            results.append({
                "Prompt": prompt,
                "Temperature": temperature,
                "Top_K": top_k,
                "Generated_Text": generated_text
            })

# Convert results into a DataFrame
df = pd.DataFrame(results)

# Save results as CSV
df.to_csv(
    "dataset/text_generation_results.csv",
    index=False
)

# Display results
print("\nText generation completed successfully!")

print("\nResults saved to:")
print("dataset/text_generation_results.csv")

print("\nGenerated Results:")
print(df)