line = ''
with open('../lr_decay_test/evaluations_decay.txt', 'r') as f:
    line = f.readline()

line = line.split('Trial:')
line = [l +'\n' for l in line]
l = 'Trial:'.join(line)
print(l)
with open('../lr_decay_test/evaluations_decay.txt', 'w') as f:
    # for l in line:
        f.write(l)
