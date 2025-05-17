import json
import time

import tqdm
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://114.212.85.164:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

shot1 = """Text: <text>Problem:  A baker is preparing cookies and needs to scale a recipe that originally requires 49 grams of sugar. If they want to make one-third more of the recipe than intended, how many grams of sugar will they need in total?  Solution Steps:  1. Calculate one-third of the original sugar amount:     \\( \\frac{49}{3} \\approx 16.33 \\) grams.  2. Add this to the original amount:     \\( 49 + 16.33 = 65.33 \\) grams.  Answer:  The baker will need 65.33 grams of sugar in total.  ---Explanation:  This problem uses the equations \\( \\frac{49}{3} \\approx 16.33 \\) (to find one-third of 49) and \\( 49 + 16.33 = 65.33 \\) (to find the total after scaling the recipe). The context of scaling a recipe provides a practical scenario for applying these calculations.</text>
Answer: <ans>Let's think step by step. One-third of the original sugar amount is 49 / 3 = 16.33 grams. Add this to the original amount is 49 + 16.33 = 65.33 grams. #### 65.33</ans>"""

shot2 = """Text: <text>Math Problem:  Maria has $95 to divide among four expenses. She spends half of the amount on rent, twice $15 on groceries, $15 on utilities, and the remaining amount on savings. How much does she save?---Step-by-Step Solution:  1. Rent Calculation:     Maria spends half of $95 on rent:     \\[   \\frac{95}{2} = 47.5 \\quad \\text{(Rent)}   \\]  2. Groceries Calculation:     She spends twice $15 on groceries:     \\[   15 \\times 2 = 30 \\quad \\text{(Groceries)}   \\]  3. Utilities Expense:     The utilities cost is $15:     \\[   15 \\quad \\text{(Utilities)}   \\]  4. Total Spent on Rent, Groceries, and Utilities:     \\[   47.5 + 30 + 15 = 92.5   \\]  5. Savings Calculation:     Subtract the total expenses from the initial amount to find the savings:     \\[   95 - 92.5 = 2.5   \\]  Answer: Maria saves $\\boxed{2.50}$.</text>
Answer: <ans>Let's think step by step. Maria spends 95 / 2 = 47.5 on rent. Maria spends 15 * 2 = 30 on groceries, and 15 = 15 on utilities. Maria saves 95 - 47.5 - 30 - 15 = 2.5. #### 2.5</ans>"""

shot3 = """Text: <text>Problem:  A magical plant starts with 2 leaves. Every day, the number of leaves triples. After two days, a magical potion is used that multiplies the leaf count by 55. How many leaves does the plant have after these transformations?Step-by-Step Solution:  1. Start with 2 leaves.  2. After the first day:     \\(2 \\times 3 = 6\\) leaves.  3. After the second day:     \\(6 \\times 3 = 18\\) leaves.  4. After applying the potion (multiply by 55):     \\(18 \\times 55 = 990\\) leaves.  Answer: The plant has \\(\\boxed{990}\\) leaves.</text>
Answer: <ans>Let's think step by step. After the first day, the plant has 2 * 3 = 6 leaves. After the second day, the plant has 6 * 3 = 18 leaves. After applying the potion, the plant finally has 18 * 55 = 990 leaves. #### 990</ans>"""

shot4 = """Text: <text>Math Problem:  A gardener plants flowers in two different gardens. In the front garden, there are 11 flowerpots, each containing 4 flowers. In the backyard, there are 6 flowerpots, each also containing 4 flowers. How many flowers does the gardener plant in total?---Step-by-Step Solution:  1. Calculate the number of flowers in the front garden:     \\( 11 \\text{ pots} \\times 4 \\text{ flowers/pot} = 44 \\text{ flowers} \\).  2. Calculate the number of flowers in the backyard:     \\( 6 \\text{ pots} \\times 4 \\text{ flowers/pot} = 24 \\text{ flowers} \\).  3. Add the flowers from both gardens:     \\( 44 + 24 = 68 \\text{ flowers} \\).  Answer: The gardener plants a total of \\(\\boxed{68}\\) flowers.</text>
Answer: <ans>Let's think step by step. The number of flowers in the front garden is 11 * 4 = 44. The number of flowers in the backyard is 6 * 4 = 24. There are 44 + 24 = 68 flowers in total. #### 68</ans>"""

prompt_pat = """Text: <text>{}</text>
Answer: <ans>Let's think step by step. """

def get_problem_answer(texts):
    shots_str = "\n\n\n".join([shot1, shot2, shot3, shot4])
    prompts = []
    for text in texts:
        prompt = shots_str + "\n\n\n" + prompt_pat.format(text)
        # print(prompt)
        # print("@=#+$-%~" * 6)
        prompts.append(prompt)

    start = time.time()
    response = client.completions.create(
        model="Qwen2.5-7B",
        prompt=prompts,
        max_tokens=1024,
        temperature=0.3,
        top_p=0.95,
        stop="\n\n\n"
    )
    print(f"Elapsed: {time.time() - start:.2f} s.")
    ans = [ch.text for ch in response.choices]
    assert len(ans) == len(texts)
    return ans

def validation():
    pass

def main():
    with open("filtered_generated_gsm_train.json", "r") as f:
        data = json.load(f)
    print(len(data))

    texts = [da["texts"] for da in data]
    questions = [da["question"] for da in data]
    answers = []
    batch_sz = 50
    num_batch = (len(data) // batch_sz) + 1
    for i in tqdm.tqdm(range(num_batch)):
        batch_text = texts[i*batch_sz: (i+1)*batch_sz]
        ans = get_problem_answer(batch_text)
        answers.extend(ans)
    assert len(questions) == len(answers)

    train_data_ex = [{
        "question": qu, "answer": an
    } for qu, an in zip(questions, answers)]

    with open("gsm_train_ex.json", "w") as f:
        json.dump(train_data_ex, f)

if __name__ == '__main__':
    main()