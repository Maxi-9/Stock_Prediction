import sys
#Deprecated
#Will be removed in a future versions

def GetOptions():
    options={}
    defaultOption={
        "-mode":"Linear",
        "-graph":"Regular",
        "-period":"max",
        "-forecast":"5"
    }
    OptionalOptions={
        "-mode":["Linear"],
        "-graph":["None","Regular","Raw"],
        "-period":"",
        "-forecast":""
    }
    RequiredOptions={
        "-stock":""
    }

    ProgramOptions={}
    ProgramOptions.update(RequiredOptions)
    ProgramOptions.update(OptionalOptions)

    
    for i in range(len(sys.argv)):
        if ("-" in sys.argv[i]):
            if (sys.argv[i] in options.keys()):
                print("Error: duplicate option "+sys.argv[i])
                return ("Error")
            else:
                if (sys.argv[i] in ProgramOptions.keys()):
                    if (sys.argv[i+1] in ProgramOptions[sys.argv[i]] or ProgramOptions[sys.argv[i]]==""):
                        options[sys.argv[i]]=sys.argv[i+1]
                    else:
                        print("Error: Not an option for "+sys.argv[i]+". The Options are:")
                        printOut="\t["
                        for k in ProgramOptions[sys.argv[i]]:
                            printOut+=k+", "
                        printOut+="]"
                        print("\t"+printOut)
                        return ("Error")
                else:
                    #option not found
                    print("Error: Not an option. The Options are:")
                    for j in ProgramOptions.keys():
                        printOut=i+" ["
                        for k in ProgramOptions[j]:
                            printOut+=k+", "
                        printOut+="]"
                        print("\t"+printOut)
                    return ("Error")
    Missing=False
    for i in defaultOption.keys():
        if not (i in options):
            options[i]=defaultOption[i]
    
    for i in RequiredOptions.keys():
        if not (i in options):
            print("Missing Required program argument: "+i)
            Missing=True
    if Missing:
        return "Error"
    
    if options["-period"]!="max":
        options["-period"]+="d"
    options["-forecast"]=int(options["-forecast"])
    return options