# -*- coding: utf-8 -*-

manual_shot_list = \
[[
"Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
"def solution():\n    \"\"\"Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\"\"\"\n    money_initial = 23\n    bagels = 5\n    bagel_cost = 3\n    money_spent = bagels * bagel_cost\n    money_left = money_initial - money_spent\n    result = money_left\n    return result"
],[
"Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
"def solution():\n    \"\"\"Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\"\"\"\n    golf_balls_initial = 58\n    golf_balls_lost_tuesday = 23\n    golf_balls_lost_wednesday = 2\n    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday\n    result = golf_balls_left\n    return result"
],[
"Jeff bought 6 pairs of shoes and 4 jerseys for $560. Jerseys cost 1/4 price of one pair of shoes. Find the shoe's price total price.",
"def solution():\n    \"\"\"Jeff bought 6 pairs of shoes and 4 jerseys for $560. Jerseys cost 1/4 price of one pair of shoes. Find the shoe's price total price.\"\"\"\n    shoes_num = 6\n    jerseys_num = 4\n    total_cost = 560\n    shoes_cost_each = Variable()\n    jerseys_cost_each = Variable()\n    total_cost = shoes_cost_each * shoes_num + jerseys_cost_each * jerseys_num\n    jerseys_cost_each = shoes_cost_each * 1 / 4\n    shoes_cost_total = shoes_cost_each * shoes_num\n    result = shoes_cost_total\n    return result"
],[
"Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
"def solution():\n    \"\"\"Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\"\"\"\n    toys_initial = 5\n    mom_toys = 2\n    dad_toys = 2\n    total_received = mom_toys + dad_toys\n    total_toys = toys_initial + total_received\n    result = total_toys\n    return result"
],[
"Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
"def solution():\n    \"\"\"Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\"\"\"\n    jason_lollipops_initial = 20\n    lollipops_given = Variable()\n    jason_lollipops_after = 12\n    jason_lollipops_after = jason_lollipops_initial - lollipops_given\n    result = lollipops_given\n    return result"
],[
"Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
"def solution():\n    \"\"\"Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\"\"\"\n    leah_chocolates = 32\n    sister_chocolates = 42\n    total_chocolates = leah_chocolates + sister_chocolates\n    chocolates_eaten = 35\n    chocolates_left = total_chocolates - chocolates_eaten\n    result = chocolates_left\n    return result"
],[
"If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
"def solution():\n    \"\"\"If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\"\"\"\n    cars_initial = 3\n    cars_arrived = 2\n    total_cars = cars_initial + cars_arrived\n    result = total_cars\n    return result"
],[
"There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
"def solution():\n    \"\"\"There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\"\"\"\n    trees_initial = 15\n    trees_planted = Variable()\n    trees_after = 21\n    trees_after = trees_initial + trees_planted\n    result = trees_planted\n    return result"
]
]
manual_shots = [
    {"question": item[0], "answer": item[1]}
    for item in manual_shot_list
]