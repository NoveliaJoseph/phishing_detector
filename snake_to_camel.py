import re

def snake_to_camel(snake_str):
    # The replacement function takes a match object and returns the uppercased letter
    def to_camel_case(match):
        return match.group(1).upper()
    
    # Match an underscore followed by a lowercase letter
    return re.sub(r'_([a-z])', to_camel_case, snake_str)

if __name__ == '__main__':
    print(snake_to_camel("my_super_cool_variable")) 
    # Output: mySuperCoolVariable
    print(snake_to_camel("alreadyCamelCase"))
    # Output: alreadyCamelCase
