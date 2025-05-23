#test

rtools_bin <- "C:/rtools40/usr/bin"  
Sys.setenv(PATH = paste(rtools_bin, Sys.getenv("PATH"), sep=";"))

system("make --version")