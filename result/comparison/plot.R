#!/usr/bin/Rscript

rm(list=ls())

library(ggplot2)
library(lubridate)

# SET PATH
getwd()
setwd("C:/Users/hkko/Desktop/fig/comparison")
# file.path(getwd(), "fig", "comparison")

# DATA READ & PREPROCESSING
read_preproc = function(len, alpha) {
  # READ DATA
  if(alpha==200) {
    d1 = read.table(file="pumap_alpha200_ops500_eps4.txt", header=TRUE, sep="\t")
    d2 = read.table(file="pumap_alpha200_ops700_eps4.txt", header=TRUE, sep="\t")
    d3 = read.table(file="pumap_alpha200_ops1000_eps4.txt", header=TRUE, sep="\t")
    d4 = read.table(file="umap_alpha200_eps4.txt", header=TRUE, sep="\t")
  } else if (alpha == 666) {
    d1 = read.table(file="pumap_alpha666_ops500_eps5.txt", header=TRUE, sep="\t")
    d2 = read.table(file="pumap_alpha666_ops700_eps5.txt", header=TRUE, sep="\t")
    d3 = read.table(file="pumap_alpha666_ops1000_eps5.txt", header=TRUE, sep="\t")
    d4 = read.table(file="umap_alpha666_eps5.txt", header=TRUE, sep="\t")
  } else {
    stop("Check alpha value")
  }

    # GET MIN LENGTH & SIZE FITTING
  min_len = min(length(d1[, 1]), length(d2[, 1]), length(d3[, 1]), length(d4[, 1]))

  if(len > min_len) {
    len = min_len
  }
  
  d1 = d1[2:len, ]
  d2 = d2[2:len, ]
  d3 = d3[2:len, ]
  d4 = d4[2:len, ]
  
  # LABELING
  d1[, "label"] = "pumap_ops500"
  d2[, "label"] = "pumap_ops700"
  d3[, "label"] = "pumap_ops1000"
  d4[, "label"] = "umap"

  # ROW BIND
  dall = rbind(d1,d2,d3,d4)
  dall$label = as.character(dall$label)
  
  # SET TIME AS POSIXct
  #options(digits.secs = 10)
  #x = strptime(dall$time_taken, format='%H:%M:%OS')
  #dall$time_taken = format(as.POSIXct(x), '%H:%M:%OS')
  
  dall$time_taken = hms(dall$time_taken)
  dall$time_taken = as.numeric(dall$time_taken)
  
  return(dall)
}

# FUNCTION TO SELECT DATA
set_data = function(data, names) {
  # CHECK CLASS FOR EACH COLUMN
  # sapply(data, class)
  
  mold = data.frame(size=as.integer(), self.epochs=as.integer(),
                    time_taken=as.numeric(), cost=as.numeric(), label=as.character()) 

  for(name in names) {
    print(name)
    mold = rbind(mold, data[which(data$label == name), ])
  }
  
  return(mold)
}


# MAIN FUNCTION
main = function() {
  data = read_preproc(len=300, alpha=200)

  # names = list("umap", "pumap_ops1000")
  names = list("umap", "pumap_ops500", "pumap_ops700", "pumap_ops1000")
  selected_data = set_data(data, names)
  
  # VISUALIZING LINE CHART
  ggplot(data=selected_data, aes(x=time_taken, y=cost, group=label, color=label)) +
    geom_line(size=1.0) + geom_point(size=1.5)+
    theme_minimal() +
    ggtitle("Progressive UMAP (PUMAP) vs. UMAP cost / time line chart") +
    xlab("time taken (sec)") +
    scale_x_continuous(breaks = round(seq(0, max(selected_data$time_taken), by = 10),1)) +
    theme(legend.title = element_text(size=12, color = "black", face="bold"),
          legend.justification=c(1,0), 
          legend.position=c(0.95, 0.70),  
          legend.background = element_blank(),
          legend.key = element_blank())

}


if (sys.nframe() == 0){
  main()
}


