import re

def is_valid_ipv4(ip):
    # Matches numbers 0-255
    segment = r'(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])'
    # Combine 4 segments separated by dots
    pattern = rf'^{segment}\.{segment}\.{segment}\.{segment}$'
    
    return bool(re.match(pattern, ip))

if __name__ == '__main__':
    print(f"192.168.1.1: {is_valid_ipv4('192.168.1.1')}")  # True
    print(f"256.0.0.1: {is_valid_ipv4('256.0.0.1')}")    # False
    print(f"192.168.01.1: {is_valid_ipv4('192.168.01.1')}") # False (leading zero)
