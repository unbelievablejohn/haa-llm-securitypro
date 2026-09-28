print("hello HAA")

name = "你的名字"
skills = ["python", "llm"]
profile = {"name": name, "week": 1, "skills": skills}

for key, value in profile.items():
    print(key, ":", value)

if profile["week"] == 1:
    print("从第1周开始")

def greet(name):
    return f"你好，{name}，开始学AI安全。"

print(greet(name))