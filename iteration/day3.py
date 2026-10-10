def check_age(age):
    if 0 <= age <= 120:
        return True
    else:
        return False

def check_confidence(score):
    if 0 <= score <= 1:
        return True
    else:
        return False

def make_decision(confidence):
    if confidence >= 0.8:
        return "EXECUTE"
    elif confidence >= 0.5:
        return "VERIFY"
    else:
        return "ASK HUMAN"


def batch_test():
    test_cases = [{"age": 22, "confidence": 0.92},
    {"age": 45, "confidence": 0.61},
    {"age": 19, "confidence": 0.33},
    {"age": 130, "confidence": 0.88},
    {"age": 56, "confidence": 1.2},
    {"age": 7, "confidence": 0.96},
    {"age": 88, "confidence": 0.52},
    {"age": 35, "confidence": 0.41},
    {"age": -5, "confidence": 0.75},
    {"age": 62, "confidence": 1.05}]
        
    

    print("===== Batch Test Start =====")
    for case in test_cases:
        age = case["age"]
        conf = case["confidence"]

        age_ok = check_age(age)
        conf_ok = check_confidence(conf)

        if age_ok and conf_ok:
            res = make_decision(conf)
            print(f"Age:{age}, Conf:{conf} | Result: {res}")
        else:
            print(f"Age:{age}, Conf:{conf} | Input Data Invalid")

    print("===== Batch Test Finished =====")


def main():
    while True:
        print("\n1、Manual single test")
        print("2、Batch auto test")
        print("3、Exit program")
        op = input("Please select function number: ")

        if op == "1":
            try:
                age = int(input("Enter age: "))
                if not check_age(age):
                    print("Age out of range 0‑120")
                    continue
                conf = float(input("Enter confidence(0‑1): "))
                if not check_confidence(conf):
                    print("Confidence out of range 0‑1")
                    continue
                decision = make_decision(conf)
                print(f"Final Decision: {decision}")
            except ValueError:
                print("Wrong input, must input number!")

        elif op == "2":
            batch_test()

        elif op == "3":
            print("Program exit.")
            break

        else:
            print("Invalid option, choose 1‑3")


if __name__ == "__main__":
    main()
