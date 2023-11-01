import random

PERCENT_DEAD = 7.0
CUT = 64_000

with open("american_bankruptcy.csv", "r") as file:
    lines = file.readlines()

header = lines[0]
instances = lines[1:]

total_instance_count = len(instances)
dead_instance_count = int(float(total_instance_count) * PERCENT_DEAD / 100.0)  # Not exact
alive_instance_count = total_instance_count - dead_instance_count

print("Total: ", total_instance_count)
print("Dead: ", dead_instance_count)
print("Alive: ", alive_instance_count)

random.shuffle(instances)

cut = 0
new_instances = []

for instance in instances:
    tokens = instance.split(",")

    if tokens[1] == "alive":
        cut += 1
        if cut >= CUT:
            new_instances.append(instance)
    elif tokens[1] == "failed":
        new_instances.append(instance)
    else:
        raise RuntimeError("What")

print("New: ", len(new_instances))

with open("american_bankruptcy_filtered.csv", "w") as file:
    file.write(header)
    file.writelines(new_instances)
