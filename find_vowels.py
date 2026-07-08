import re

def find_vowel_positions(text):
    for match in re.finditer(r'[aeiouAEIOU]', text):
        print(f"Vowel '{match.group()}' found at index {match.start()}")

if __name__ == '__main__':
    find_vowel_positions("Antigravity")
