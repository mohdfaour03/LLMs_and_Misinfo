
try:
    with open('stats.txt', 'r', encoding='utf-16le') as f:
        print(f.read())
except:
    try:
        with open('stats.txt', 'r', encoding='utf-8') as f:
            print(f.read())
    except Exception as e:
        print(e)
