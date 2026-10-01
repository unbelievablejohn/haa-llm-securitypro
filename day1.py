print("Hello,confidence and safety!")
print("I am starting from zero.")
name = "dragon"
age=18
research_direction = "AI confidently_and_safely_output"
print(name)
print(age)
print(research_direction)
user_name = input("请输入你的名字")
print(user_name)
age_input = int(input("请输入你的年龄："))
if age_input >= 18:
    print("You are an adult")
else:
    print("You are underage") 
confidence_score = float(input("Please inputAI confidence score(0-1):"))
if confidence_score >= 0.8:
    print("Confidence is high,safe to respond.")
elif confidence_score >= 0.5:
    print("Confidence is medium,need double_check.")
else:
    print("Confidence is too low,refuse to refuse.")