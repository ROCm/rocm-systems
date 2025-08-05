def sample_ex(source_str):
    spm_dump = source_str

    file_in = open(spm_dump, "r")

    #extract valid sample data size
    line = file_in.readline()
    line.strip()

    size = int(line, 16)

    line = file_in.readline()
    line.strip()

    size += (int(line, 16) << 16)

    print("valid sampl buffer size: %d"%(size))

    file_out = open("samples.txt", "w")

    line_no = 2

    non_zero_samples = 0

    while (size > 0):
        line = file_in.readline()
        line.strip()
        if line_no >= 16:
            file_out.write(line)
            size -= 2
            if int(line, 16) != 0:
                non_zero_samples += 1
                #print("%d: sample: %s"%(line_no, line))
        line_no += 1


    print ("non_zero_samples: %d"%(non_zero_samples))

    file_in.close()
    file_out.close()
