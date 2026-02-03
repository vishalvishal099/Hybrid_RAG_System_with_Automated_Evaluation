"""
Create Evaluation Dataset with Ground Truth
Generates question-URL pairs and expected answers for evaluation
"""

import json
from pathlib import Path


def create_evaluation_dataset():
    """
    Create ground truth evaluation dataset
    Returns list of test cases with questions, URLs, and expected answers
    """
    
    # Sample evaluation questions with ground truth
    test_cases = [
        {
            "question": "What is artificial intelligence?",
            "ground_truth_url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "ground_truth_urls": ["https://en.wikipedia.org/wiki/Artificial_intelligence"],
            "ground_truth_answer": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. It is a field of study in computer science that develops and studies methods and software which enable machines to perceive their environment and uses learning and intelligence to take actions that maximize their chances of achieving defined goals."
        },
        {
            "question": "Who invented the telephone?",
            "ground_truth_url": "https://en.wikipedia.org/wiki/Alexander_Graham_Bell",
            "ground_truth_urls": [
                "https://en.wikipedia.org/wiki/Alexander_Graham_Bell",
                "https://en.wikipedia.org/wiki/Telephone"
            ],
            "ground_truth_answer": "Alexander Graham Bell invented the telephone. He was awarded the first US patent for the invention of the telephone in 1876."
        },
        {
            "question": "When was the Roman Empire founded?",
            "ground_truth_url": "https://en.wikipedia.org/wiki/Roman_Empire",
            "ground_truth_urls": ["https://en.wikipedia.org/wiki/Roman_Empire"],
            "ground_truth_answer": "The Roman Empire was founded in 27 BC when Augustus became the first Roman emperor, marking the end of the Roman Republic."
        },
        {
            "question": "What is the capital of France?",
            "ground_truth_url": "https://en.wikipedia.org/wiki/Paris",
            "ground_truth_urls": [
                "https://en.wikipedia.org/wiki/Paris",
                "https://en.wikipedia.org/wiki/France"
            ],
            "ground_truth_answer": "Paris is the capital of France. It is the country's largest city and has been the capital since the 12th century."
        },
        {
            "question": "What is photosynthesis?",
            "ground_truth_url": "https://en.wikipedia.org/wiki/Photosynthesis",
            "ground_truth_urls": ["https://en.wikipedia.org/wiki/Photosynthesis"],
            "ground_truth_answer": "Photosynthesis is the process by which plants and other organisms convert light energy into chemical energy that can be used to fuel the organism's activities. Plants use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of sugar."
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "ground_truth_url": "https://en.wikipedia.org/wiki/William_Shakespeare",
            "ground_truth_urls": [
                "https://en.wikipedia.org/wiki/William_Shakespeare",
                "https://en.wikipedia.org/wiki/Romeo_and_Juliet"
            ],
            "ground_truth_answer": "William Shakespeare wrote Romeo and Juliet. It is one of his most famous tragedies, written between 1594 and 1596."
        },
        {
            "question": "What is the speed of light?",
            "ground_truth_url": "https://en.wikipedia.org/wiki/Speed_of_light",
            "ground_truth_urls": ["https://en.wikipedia.org/wiki/Speed_of_light"],
            "ground_truth_answer": "The speed of light in vacuum is exactly 299,792,458 meters per second (approximately 300,000 kilometers per second or 186,000 miles per second). It is denoted by the symbol c."
        },
        {
            "question": "What is DNA?",
            "ground_truth_url": "https://en.wikipedia.org/wiki/DNA",
            "ground_truth_urls": ["https://en.wikipedia.org/wiki/DNA"],
            "ground_truth_answer": "DNA (deoxyribonucleic acid) is a molecule that carries genetic instructions for the development, functioning, growth and reproduction of all known organisms. It consists of two strands that coil around each other to form a double helix."
        },
        {
            "question": "Who painted the Mona Lisa?",
            "ground_truth_url": "https://en.wikipedia.org/wiki/Leonardo_da_Vinci",
            "ground_truth_urls": [
                "https://en.wikipedia.org/wiki/Leonardo_da_Vinci",
                "https://en.wikipedia.org/wiki/Mona_Lisa"
            ],
            "ground_truth_answer": "Leonardo da Vinci painted the Mona Lisa. It is an oil painting created between 1503 and 1519, and is one of the most famous paintings in the world."
        },
        {
            "question": "What is the largest planet in our solar system?",
            "ground_truth_url": "https://en.wikipedia.org/wiki/Jupiter",
            "ground_truth_urls": ["https://en.wikipedia.org/wiki/Jupiter"],
            "ground_truth_answer": "Jupiter is the largest planet in our solar system. It is a gas giant with a mass more than two and a half times that of all the other planets in the solar system combined."
        }
    ]
    
    return test_cases


def save_evaluation_dataset(output_path: str = "data/evaluation_dataset.json"):
    """Save evaluation dataset to JSON file"""
    dataset = create_evaluation_dataset()
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'test_cases': dataset,
            'num_cases': len(dataset),
            'description': 'Ground truth evaluation dataset with questions, URLs, and expected answers'
        }, f, indent=2)
    
    print(f"✓ Saved {len(dataset)} evaluation test cases to {output_path}")
    return dataset


def load_evaluation_dataset(path: str = "data/evaluation_dataset.json"):
    """Load evaluation dataset from JSON file"""
    with open(path, 'r') as f:
        data = json.load(f)
    return data['test_cases']


if __name__ == "__main__":
    # Create and save the dataset
    dataset = save_evaluation_dataset()
    
    print(f"\nDataset Summary:")
    print(f"  Total test cases: {len(dataset)}")
    print(f"\nSample test case:")
    print(f"  Question: {dataset[0]['question']}")
    print(f"  URL: {dataset[0]['ground_truth_url']}")
    print(f"  Answer: {dataset[0]['ground_truth_answer'][:100]}...")
