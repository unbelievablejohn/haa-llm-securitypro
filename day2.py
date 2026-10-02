def check_confidence(score):
    """Verify confidence between 0 and 1"""
    if 0 <= score <= 1:
        return True
    return False


def check_age(age):
    """Verify age between 0 and 120"""
    if 0 <= age <= 120:
        return True
    return False

def make_decision(confidence):
    if confidence >= 0.8:
        return "EXECUTE"
    elif confidence >= 0.5:
        return "VERIFY"
    else:
        return "ASK HUMAN"
def main():
    print("===== Prohibited Word Detector =====")
    while True:
        print("\n1、Start detection\n2、Exit program")
        op = input("Please enter your option number: ")

        if op == "2":
            print("Program terminated")
            break
        elif op == "1":
            # Catch input‑related exceptions
            try:
                age = int(input("Please enter age: "))
                if not check_age(age):
                    print("Age must be between 0‑120!")
                    continue

                conf = float(input("Please enter confidence (0‑1): "))
                if not check_confidence(conf):
                    print("Confidence must range from 0 to 1!")
                    continue

                print(f"Input check passed, Age:{age}, Confidence:{conf}")
                result = make_decision(conf)
                print(f"Final Decision:{result}")

            except ValueError:
                print("Invalid input format! Please enter valid numbers")
        else:
            print("Invalid choice, please type 1 or 2")


if __name__ == "__main__":
    main()
