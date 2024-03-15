# Created on Nov 28 10:47:58 2023

# @author: LJ

library("factoextra")

## for all enviroments,change save below
# df <- read.csv("pca_data_all.csv")
# df.pr <- prcomp(df[c(4:11)], center = TRUE, scale = TRUE)
# summary(df.pr)

df_all <- read.csv("pca_data.csv")
# Define variables
terrain <- 'Flat'
frequency <- '0'

# Create the experiment names using sprintf
lamarckism_experiment <- sprintf("Lamarckism_%s_%s", terrain, frequency)
darwinism_experiment <- sprintf("Darwinism_%s_%s", terrain, frequency)

# Filter the data based on the defined Terrain and Frequency
df <- df_all[df_all$experiment %in% c(lamarckism_experiment, darwinism_experiment), ]

df.pr <- prcomp(df[c(4:11)], center = TRUE, scale = TRUE)
summary(df.pr)

# Get the loadings
loadings <- df.pr$rotation

# Print the loadings for each principal component
for (i in 1:ncol(loadings)) {
  cat("Loadings for PC", i, ":\n")
  print(loadings[, i])
}

screeplot(df.pr, type = "l", npcs = 8, main = "Screeplot of the 8 features")
abline(h = 1, col="red", lty=5)
legend("topright", legend=c("Eigenvalue = 1"),
       col=c("red"), lty=5, cex=0.6)
cumpro <- cumsum(df.pr$sdev^2 / sum(df.pr$sdev^2))
plot(cumpro[0:15], xlab = "PC #", ylab = "Amount of explained variance", main = "Cumulative variance plot")
abline(v = 6, col="blue", lty=5)
abline(h = 0.88759, col="blue", lty=5)
legend("topleft", legend=c("Cut-off @ PC6"),
       col=c("blue"), lty=5, cex=0.6)

plot(df.pr$x[,1],df.pr$x[,2], xlab="PC1", ylab = "PC2", main = "PC1 / PC2 - plot")



pca_plot=fviz_pca_biplot(df.pr, geom.ind = "point", pointshape = 21,
                         pointsize = 1.5,
                         fill.ind = df$experiment,
                         col.ind = "black",
                         palette = "jco",
                         #addEllipses = TRUE,

                         addEllipses = TRUE, ellipse.type = "convex",
                         ellipse.alpha = 0.07,
                         # ellipse.linetype = 2, ellipse.fill = NA,

                         alpha.var ="contrib", col.var = "contrib",
                         gradient.cols = "Greens",

                         legend.title = list(fill = "Dataset", color = "Contrib",
                                             alpha = "Contrib")) +
  ggtitle("PCA biplot from 8 morphological traits") +
  theme(plot.title = element_text(hjust = 0.5,size = 16),
        legend.text = element_text(size = 14),
        axis.text = element_text(size = 14),
        axis.title = element_text(size = 14))  # Set the desired font size here


# Save the plot
ggsave(sprintf("pca_biplot_%s_%s.png", terrain, frequency), pca_plot, height = 8, width = 11, bg = "white")