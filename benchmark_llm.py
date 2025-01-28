import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import re
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Load your benchmark dataset
df = pd.read_csv('ecologic_benchmark.csv')

# Initialize the LLM (using Falcon-7B as example)
model_name = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    #trust_remote_code=True,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

def parse_answer(response):
    """Extract a/b/c answer from model response"""
    response = response.lower()
    match = re.search(r'\b(a|b|c)[).:]?\b', response)
    if match:
        return match.group(1)
    if 'increase' in response:
        return 'a'
    elif 'decrease' in response:
        return 'b'
    elif 'no change' in response:
        return 'c'
    return None

def get_llm_answer(prompt):
    """Get answer from LLM with error handling"""
    try:
        response = pipe(
            prompt,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )[0]['generated_text']
        return parse_answer(response.split(prompt)[-1].strip())
    except Exception as e:
        print(f"Error processing prompt: {e}")
        return None

# Initialize result columns
for col in ['correct_a', 'correct_b', 'correct_c']:
    df[col] = 0

# Map expected answers from Change column
answer_map = {1: 'a', -1: 'b', 0: 'c'}
df['expected'] = df['Change'].map(answer_map)

# Process all rows
for idx, row in tqdm(df.iterrows(), total=len(df)):
    # Generate prompts
    prompts = {
        'a': f"{row['Prompt_A']} If the population of {row['Intervention']} increases. What happens to the population of {row['Target']}? Does it a) increase b) decrease or is there c) no change?",
        'b': f"{row['Prompt_B']} If the population of {row['Intervention_pseudonym']} increases. What happens to the population of {row['Target_pseudonym']}? Does it a) increase b) decrease or is there c) no change?",
        'c': f"{row['Prompt_C']} form a Food-Web. If {row['Intervention']} increases. What happens to the population of {row['Target']}? Does it a) increase b) decrease or is there c) no change?"
    }
    
    # Get and evaluate responses
    for prompt_type in ['a', 'b', 'c']:
        answer = get_llm_answer(prompts[prompt_type])
        df.at[idx, f'correct_{prompt_type}'] = int(answer == row['expected'])

# Calculate overall accuracy
accuracies = {
    'Prompt A': df['correct_a'].mean(),
    'Prompt B': df['correct_b'].mean(),
    'Prompt C': df['correct_c'].mean()
}
print("\nOverall Accuracy:")
for prompt, acc in accuracies.items():
    print(f"{prompt}: {acc:.2%}")

# Visualization functions
def plot_accuracy_by(feature):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df.melt(id_vars=[feature], 
                            value_vars=['correct_a', 'correct_b', 'correct_c'],
                            var_name='Prompt',
                            value_name='Accuracy'),
                x=feature, y='Accuracy', hue='Prompt')
    plt.title(f'Accuracy by {feature}')
    plt.ylim(0, 1)
    plt.show()

# Generate plots
plot_accuracy_by('#Nodes')
plot_accuracy_by('#Edges')
plot_accuracy_by('Depth')