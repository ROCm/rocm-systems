import struct

def sample_ex(source_str, length):
    file = open(source_str, "rb")
    file_out = open("samples.txt", "w")

    count = 0
    sample = file.read(2)
    while sample:
        if length != -1:
            if count >= length:
                break
        #print(sample)
        value = struct.unpack('H', sample)[0]
        t = "{:04x}".format(value)
        #print(t)
        line = t + "\n"
        file_out.write(line)
        sample = file.read(2)
        count += 2

    file.close()
    file_out.close()
