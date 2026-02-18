"""
Dataset Loader from HuggingFace Hub
Prepares benchmark datasets for uncertainty evaluation
"""

import json
import random
from typing import List, Dict, Optional
from datasets import load_dataset
import numpy as np

random.seed(42)
np.random.seed(42)


class HFDatasetLoader:
    """Load and prepare datasets from HuggingFace Hub"""
    
    def __init__(self):
        """Initialize dataset loader"""
        self.datasets = {}
    
    def load_scifact(self, n_samples: int = 20) -> List[Dict]:
        """
        Load SciFact dataset (scientific fact verification)
        This is what you used in your notebook!
        
        Args:
            n_samples: Number of samples to load
        
        Returns:
            List of question dictionaries
        """
        print(f"Loading SciFact dataset ({n_samples} samples)...")
        
        try:
            # Load dataset
            dataset = load_dataset("mteb/scifact", split="test")
            
            questions = []
            for i, item in enumerate(dataset):
                if len(questions) >= n_samples:
                    break
                
                # SciFact has claims and labels
                claim_id = item.get('_id', i)
                claim_text = item.get('text', '')
                
                if not claim_text:
                    continue
                
                questions.append({
                    'id': len(questions),
                    'question': claim_text,
                    'answer': 'Supported' if item.get('label', 0) == 1 else 'Refuted',
                    'category': 'answerable',
                    'evidence_quality': 'clear',
                    'source': 'scifact',
                    'original_id': claim_id
                })
            
            print(f"  Loaded {len(questions)} samples from SciFact")
            return questions
            
        except Exception as e:
            print(f"  Error loading SciFact: {e}")
            return self._get_fallback_science(n_samples)
    
    def load_natural_questions(self, n_samples: int = 20) -> List[Dict]:
        """
        Load Natural Questions dataset
        
        Args:
            n_samples: Number of samples to load
        
        Returns:
            List of question dictionaries
        """
        print(f"Loading Natural Questions ({n_samples} samples)...")
        
        try:
            # Load NQ-open (open domain QA)
            dataset = load_dataset("nq_open", split="validation")
            
            questions = []
            sampled_indices = random.sample(range(len(dataset)), min(n_samples * 3, len(dataset)))
            
            for idx in sampled_indices:
                if len(questions) >= n_samples:
                    break
                
                item = dataset[idx]
                question = item['question']
                answers = item['answer']
                
                # Only include if answer is short (factoid)
                if isinstance(answers, list) and len(answers) > 0:
                    answer = answers[0]
                    if len(answer.split()) <= 6:
                        questions.append({
                            'id': len(questions),
                            'question': question,
                            'answer': answer,
                            'category': 'answerable',
                            'evidence_quality': 'clear',
                            'source': 'natural_questions'
                        })
            
            print(f"  Loaded {len(questions)} samples from Natural Questions")
            return questions
            
        except Exception as e:
            print(f"  Error loading Natural Questions: {e}")
            return self._get_fallback_factual(n_samples)
    
    def load_trivia_qa(self, n_samples: int = 20) -> List[Dict]:
        """
        Load TriviaQA dataset
        
        Args:
            n_samples: Number of samples to load
        
        Returns:
            List of question dictionaries
        """
        print(f"Loading TriviaQA ({n_samples} samples)...")
        
        try:
            dataset = load_dataset("trivia_qa", "unfiltered.nocontext", split="validation")
            
            questions = []
            sampled_indices = random.sample(range(len(dataset)), min(n_samples * 2, len(dataset)))
            
            for idx in sampled_indices:
                if len(questions) >= n_samples:
                    break
                
                item = dataset[idx]
                question = item['question']
                answer = item['answer']['value']
                
                if len(answer.split()) <= 6:
                    questions.append({
                        'id': len(questions),
                        'question': question,
                        'answer': answer,
                        'category': 'answerable',
                        'evidence_quality': 'clear',
                        'source': 'trivia_qa'
                    })
            
            print(f"  Loaded {len(questions)} samples from TriviaQA")
            return questions
            
        except Exception as e:
            print(f"  Error loading TriviaQA: {e}")
            return self._get_fallback_trivia(n_samples)
    
    def load_popqa(self, n_samples: int = 20) -> List[Dict]:
        """
        Load PopQA (long-tail entity questions)
        
        Args:
            n_samples: Number of samples to load
        
        Returns:
            List of question dictionaries
        """
        print(f"Loading PopQA ({n_samples} samples)...")
        
        try:
            dataset = load_dataset("akariasai/PopQA", split="test")
            
            questions = []
            
            # Sample from long-tail (low popularity entities)
            long_tail = [item for item in dataset if item.get('s_pop', 0) < 1000]
            sampled = random.sample(long_tail, min(n_samples, len(long_tail)))
            
            for item in sampled:
                questions.append({
                    'id': len(questions),
                    'question': item['question'],
                    'answer': item['possible_answers'][0] if item.get('possible_answers') else item.get('obj', ''),
                    'category': 'weak_evidence',
                    'evidence_quality': 'weak',
                    'source': 'popqa',
                    'entity_popularity': item.get('s_pop', 0)
                })
            
            print(f"  Loaded {len(questions)} samples from PopQA")
            return questions
            
        except Exception as e:
            print(f"  Error loading PopQA: {e}")
            return self._get_fallback_longtail(n_samples)
    
    def create_conflicting_evidence(self, n_samples: int = 20) -> List[Dict]:
        """
        Create questions with conflicting/contradictory evidence
        
        Args:
            n_samples: Number of samples to create
        
        Returns:
            List of question dictionaries
        """
        print(f"Creating {n_samples} conflicting evidence questions...")
        
        conflicting_qa = [
            {
                'question': 'What is the capital of Turkey?',
                'answer': 'Ankara',
                'conflicting_info': 'Istanbul is often mistaken as the capital',
                'category': 'conflicting_evidence'
            },
            {
                'question': 'Who invented the light bulb?',
                'answer': 'Thomas Edison',
                'conflicting_info': 'Joseph Swan also developed the light bulb independently',
                'category': 'conflicting_evidence'
            },
            {
                'question': 'What is the tallest mountain in the world?',
                'answer': 'Mount Everest',
                'conflicting_info': 'Mauna Kea is taller when measured from base to peak',
                'category': 'conflicting_evidence'
            },
            {
                'question': 'How many planets are in our solar system?',
                'answer': '8',
                'conflicting_info': 'Pluto was previously considered the 9th planet',
                'category': 'conflicting_evidence'
            },
            {
                'question': 'What is the longest river in the world?',
                'answer': 'Nile River',
                'conflicting_info': 'The Amazon River is sometimes cited as longer',
                'category': 'conflicting_evidence'
            },
            {
                'question': 'Who painted the Mona Lisa?',
                'answer': 'Leonardo da Vinci',
                'conflicting_info': 'Some theories suggest it was a collaborative work',
                'category': 'conflicting_evidence'
            },
            {
                'question': 'What is the speed of light?',
                'answer': '299,792,458 meters per second',
                'conflicting_info': 'Often approximated as 300,000 km/s',
                'category': 'conflicting_evidence'
            },
            {
                'question': 'What year did World War II end?',
                'answer': '1945',
                'conflicting_info': 'Peace treaties were signed in 1946',
                'category': 'conflicting_evidence'
            },
            {
                'question': 'Who was the first person to walk on the Moon?',
                'answer': 'Neil Armstrong',
                'conflicting_info': 'Buzz Aldrin followed shortly after',
                'category': 'conflicting_evidence'
            },
            {
                'question': 'What is the largest ocean on Earth?',
                'answer': 'Pacific Ocean',
                'conflicting_info': 'By volume, the Pacific is larger than all land area combined',
                'category': 'conflicting_evidence'
            },
            {
                'question': 'What is the smallest country in the world?',
                'answer': 'Vatican City',
                'conflicting_info': 'Monaco is the smallest non-enclave sovereign state',
                'category': 'conflicting_evidence'
            },
            {
                'question': 'How many bones are in the adult human body?',
                'answer': '206',
                'conflicting_info': 'Babies are born with approximately 270 bones',
                'category': 'conflicting_evidence'
            },
            {
                'question': 'What is the largest planet in our solar system?',
                'answer': 'Jupiter',
                'conflicting_info': 'Saturn has the most extensive ring system',
                'category': 'conflicting_evidence'
            },
            {
                'question': 'What is the boiling point of water at sea level?',
                'answer': '100°C',
                'conflicting_info': 'Changes with altitude and air pressure',
                'category': 'conflicting_evidence'
            },
            {
                'question': 'What is the main component of Earth\'s atmosphere?',
                'answer': 'Nitrogen',
                'conflicting_info': 'Oxygen is more important for life despite being less abundant',
                'category': 'conflicting_evidence'
            },
            {
                'question': 'What year did the Titanic sink?',
                'answer': '1912',
                'conflicting_info': 'Some accounts incorrectly state 1913',
                'category': 'conflicting_evidence'
            },
            {
                'question': 'Who wrote Romeo and Juliet?',
                'answer': 'William Shakespeare',
                'conflicting_info': 'The story was based on earlier works by other authors',
                'category': 'conflicting_evidence'
            },
            {
                'question': 'What is the chemical symbol for gold?',
                'answer': 'Au',
                'conflicting_info': 'Derived from Latin aurum, not English gold',
                'category': 'conflicting_evidence'
            },
            {
                'question': 'How many continents are there?',
                'answer': '7',
                'conflicting_info': 'Some models combine Europe and Asia as Eurasia (6 continents)',
                'category': 'conflicting_evidence'
            },
            {
                'question': 'What is the currency of Japan?',
                'answer': 'Yen',
                'conflicting_info': 'Often confused with the Chinese Yuan',
                'category': 'conflicting_evidence'
            },
        ]
        
        questions = []
        for i, item in enumerate(conflicting_qa[:n_samples]):
            questions.append({
                'id': i,
                'question': item['question'],
                'answer': item['answer'],
                'category': 'conflicting_evidence',
                'evidence_quality': 'conflicting',
                'source': 'manual',
                'conflicting_info': item.get('conflicting_info', '')
            })
        
        print(f"  Created {len(questions)} conflicting evidence questions")
        return questions
    
    def build_60_question_benchmark(self) -> List[Dict]:
        """
        Build 60-question benchmark with balanced categories
        
        Composition:
        - 20 Answerable (clear evidence): 10 SciFact + 10 NaturalQuestions/TriviaQA
        - 20 Weak Evidence: 20 PopQA (long-tail entities)
        - 20 Conflicting Evidence: 20 manually created
        
        Returns:
            List of 60 question dictionaries
        """
        print("\n" + "="*70)
        print("Building 60-Question Benchmark from HuggingFace Hub")
        print("="*70 + "\n")
        
        # 20 Answerable (clear evidence)
        print("1. Answerable Questions (clear evidence)...")
        answerable = []
        answerable.extend(self.load_scifact(10))
        
        # Try Natural Questions, fallback to TriviaQA
        try:
            answerable.extend(self.load_natural_questions(10))
        except:
            print("  Natural Questions failed, using TriviaQA...")
            answerable.extend(self.load_trivia_qa(10))
        
        # 20 Weak Evidence
        print("\n2. Weak Evidence Questions...")
        weak_evidence = self.load_popqa(20)
        
        # 20 Conflicting Evidence
        print("\n3. Conflicting Evidence Questions...")
        conflicting = self.create_conflicting_evidence(20)
        
        # Combine all
        dataset = answerable + weak_evidence + conflicting
        
        # Re-index
        for i, item in enumerate(dataset):
            item['id'] = i
            # Add should_abstain flag
            item['should_abstain'] = (item['category'] in ['weak_evidence', 'conflicting_evidence'])
        
        print("\n" + "="*70)
        print(f"Dataset Composition:")
        print(f"  Answerable (clear evidence): {len(answerable)}")
        print(f"  Weak Evidence:               {len(weak_evidence)}")
        print(f"  Conflicting Evidence:        {len(conflicting)}")
        print(f"  Total:                       {len(dataset)}")
        print("="*70 + "\n")
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict], output_path: str = "benchmark_60_hf.json"):
        """Save dataset to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"✓ Dataset saved to {output_path}")
    
    # Fallback methods (if HF datasets fail)
    
    def _get_fallback_science(self, n: int) -> List[Dict]:
        """Fallback scientific questions"""
        questions = [
            {'question': 'Does photosynthesis require sunlight?', 'answer': 'Yes'},
            {'question': 'Is DNA the molecule that stores genetic information?', 'answer': 'Yes'},
            {'question': 'Does water freeze at 0 degrees Celsius?', 'answer': 'Yes'},
            {'question': 'Is the Earth approximately 4.5 billion years old?', 'answer': 'Yes'},
            {'question': 'Do mitochondria produce ATP?', 'answer': 'Yes'},
            {'question': 'Is gravity a fundamental force of nature?', 'answer': 'Yes'},
            {'question': 'Does the human heart have four chambers?', 'answer': 'Yes'},
            {'question': 'Is the speed of light constant in a vacuum?', 'answer': 'Yes'},
            {'question': 'Do neurons transmit electrical signals?', 'answer': 'Yes'},
            {'question': 'Is carbon essential for all known life?', 'answer': 'Yes'},
        ]
        return [{'id': i, **q, 'category': 'answerable', 'source': 'manual'} 
                for i, q in enumerate(questions[:n])]
    
    def _get_fallback_factual(self, n: int) -> List[Dict]:
        """Fallback factual questions"""
        questions = [
            {'question': 'What is the capital of France?', 'answer': 'Paris'},
            {'question': 'Who wrote "1984"?', 'answer': 'George Orwell'},
            {'question': 'What is the largest organ in the human body?', 'answer': 'Skin'},
            {'question': 'What is the chemical formula for water?', 'answer': 'H2O'},
            {'question': 'Who painted the Sistine Chapel ceiling?', 'answer': 'Michelangelo'},
            {'question': 'What is the capital of Japan?', 'answer': 'Tokyo'},
            {'question': 'What planet is known as the Red Planet?', 'answer': 'Mars'},
            {'question': 'Who discovered penicillin?', 'answer': 'Alexander Fleming'},
            {'question': 'What is the smallest prime number?', 'answer': '2'},
            {'question': 'What is the hardest natural substance?', 'answer': 'Diamond'},
        ]
        return [{'id': i, **q, 'category': 'answerable', 'source': 'manual'} 
                for i, q in enumerate(questions[:n])]
    
    def _get_fallback_trivia(self, n: int) -> List[Dict]:
        """Fallback trivia questions"""
        questions = [
            {'question': 'What is the only mammal capable of true flight?', 'answer': 'Bat'},
            {'question': 'What is the national animal of Scotland?', 'answer': 'Unicorn'},
            {'question': 'What year was the first iPhone released?', 'answer': '2007'},
            {'question': 'What is the rarest blood type?', 'answer': 'AB negative'},
            {'question': 'What is the largest desert in the world?', 'answer': 'Antarctic Desert'},
            {'question': 'What element has the atomic number 79?', 'answer': 'Gold'},
            {'question': 'What is the most widely spoken language in Brazil?', 'answer': 'Portuguese'},
            {'question': 'What year did the Berlin Wall fall?', 'answer': '1989'},
            {'question': 'What is the smallest bone in the human body?', 'answer': 'Stapes'},
            {'question': 'Who invented the World Wide Web?', 'answer': 'Tim Berners-Lee'},
        ]
        return [{'id': i, **q, 'category': 'answerable', 'source': 'manual'} 
                for i, q in enumerate(questions[:n])]
    
    def _get_fallback_longtail(self, n: int) -> List[Dict]:
        """Fallback long-tail questions"""
        questions = [
            {'question': 'What is the capital of Bhutan?', 'answer': 'Thimphu'},
            {'question': 'Who won the 2019 Cricket World Cup?', 'answer': 'England'},
            {'question': 'What is the official language of Liechtenstein?', 'answer': 'German'},
            {'question': 'Who directed the film "Parasite"?', 'answer': 'Bong Joon-ho'},
            {'question': 'What is the currency of Iceland?', 'answer': 'Icelandic króna'},
            {'question': 'Who wrote "The Kite Runner"?', 'answer': 'Khaled Hosseini'},
            {'question': 'What is the official language of Ethiopia?', 'answer': 'Amharic'},
            {'question': 'What is the capital of Mongolia?', 'answer': 'Ulaanbaatar'},
            {'question': 'Who composed the opera "Carmen"?', 'answer': 'Georges Bizet'},
            {'question': 'What is the smallest bone in the human body?', 'answer': 'Stapes'},
        ]
        return [{'id': i, **q, 'category': 'weak_evidence', 'source': 'manual'} 
                for i, q in enumerate(questions[:n])]


def main():
    """Build and save 60-question benchmark"""
    loader = HFDatasetLoader()
    dataset = loader.build_60_question_benchmark()
    loader.save_dataset(dataset, "benchmark_60_hf.json")
    
    # Print samples
    print("\n" + "="*70)
    print("Sample Questions")
    print("="*70)
    
    for category in ['answerable', 'weak_evidence', 'conflicting_evidence']:
        samples = [q for q in dataset if q['category'] == category]
        if samples:
            sample = samples[0]
            print(f"\nCategory: {category}")
            print(f"Question: {sample['question']}")
            print(f"Answer: {sample['answer']}")
            print(f"Source: {sample['source']}")


if __name__ == "__main__":
    main()