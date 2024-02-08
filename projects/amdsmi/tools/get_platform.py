import re
import argparse

def get_platforms(file_name):
  # removing the new line characters
  with open(file_name) as f:
      lines = [line.rstrip() for line in f]
  
  platform_map = {}
  platform_lines = ""
  function_line = ""
  for line in lines:
      if "@platform" in line:
          platform_lines += line.replace("*","").lstrip()
      if platform_lines =="":
          continue
      l = line.lstrip()
      if len(l) == 0:
          continue
      if l[0] == '*' or l[0]== '#' or (len(l)>1 and l[0] == '/' and l[1] == '*'):
          if function_line != "":
              func = ' '.join(function_line.split())
              platforms = re.findall(r"\{(.*?)\}", platform_lines)
              for p in platforms:
                  if p not in platform_map:
                      platform_map[p]=[func]
                  else:
                      platform_map[p].append(func)
              platform_lines = ""
          function_line = ""
          continue
      function_line += line  + " " 
  return platform_map

# string in list1 but not in list2
def diff(list1, list2):
    result = []
    for l1 in list1:
        if l1 not in list2:
            result.append(l1)
    return result

def common(list1, list2):
    result = []
    for l1 in list1:
        if l1 in list2:
            result.append(l1)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='amdsmi platform classifier')
    parser.add_argument('--platforms', default=False, action='store_true', help='list supported platforms')
    parser.add_argument('--list', default="", metavar="platform" , help='List function in a platform')
    parser.add_argument('--diff' , default=None, metavar="platform", nargs=2, help='Find the APIs in platform1 but not in platform2')
    parser.add_argument('--common' , default=None, metavar="platform", nargs=2, help='Find the common APIs of two platforms')

    file_name="../include/amd_smi/amdsmi.h"
    args = parser.parse_args()
    platform_map=get_platforms(file_name)
    if args.platforms:
        for p in platform_map:
            print (p)
        exit(0)

    if args.list:
        if args.list not in platform_map:
            print("Unknown platform ", args.list)
            exit(1)
        for f in platform_map[args.list]:
            print (f)
        exit(0)
        
    if args.diff != None:
        if args.diff[0] not in platform_map or args.diff[1] not in platform_map:
            print("Unknown platforms ", args.diff)
            exit(1)
        platforms0 = platform_map[args.diff[0]]
        platforms1 = platform_map[args.diff[1]]
        result = diff(platforms0, platforms1)
        print("APIs in", args.diff[0], "but not in", args.diff[1])
        for f in result:
            print(f)
        exit(0)
        
        
    if args.common != None:
        if args.common[0] not in platform_map or args.common[1] not in platform_map:
            print("Unknown platforms ", args.common)
            exit(1)
        platforms0 = platform_map[args.common[0]]
        platforms1 = platform_map[args.common[1]]
        result = common(platforms0, platforms1)
        print("APIs in both ", args.common[0], "and", args.common[1])
        for f in result:
            print(f)
        exit(0)
        
    parser.print_help()

