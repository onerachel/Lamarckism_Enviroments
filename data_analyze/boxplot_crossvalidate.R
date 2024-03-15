# Created on Nov 28 10:47:58 2023

# @author: LJ


library(ggplot2)
library(extrafont)

df <- read.table("best_crossvalidate.csv", header = TRUE, sep = ',')
options(warn = -1)
head(df)

df$arena = factor(df$arena)
df$robot = factor(df$robot)

p0 <- ggplot(data = df, aes(x = arena, y = fitness, fill = robot)) +
  stat_boxplot(geom = "errorbar") +
  geom_boxplot() +
  stat_summary(fun = mean, geom = "point", col = "red",
               position = position_dodge2(width = 0.75, preserve = "single")) +
  xlab(NULL) + ylab("fitness")

# Define new colors (two shades of purple and two shades of blue)
cols <- c("#6C8EBF", "skyblue3","#9673A6", "thistle3")

p <- p0 + theme_bw() +
  theme(axis.text.x = element_text(size = 25, color = "grey20", angle = 0), # Angle set to 0
        axis.text.y = element_text(size = 25, color = "grey20"),
        axis.title = element_text(size = 25),
        plot.title = element_text(size = 25, hjust = 0.5),
        legend.text = element_text(size = 17),
        legend.title = element_text(size = 17, hjust = 0.4),
        legend.key.height = unit(0.6, "cm"),
        legend.key.width = unit(0.6, "cm"),
        legend.position = "top",
        text = element_text(family = "Times New Roman", size = 12)
  ) +
  guides(fill = guide_legend(title = "Robot", nrow = 2)) + # Legend in two rows
  scale_fill_manual(values = cols)

ggsave(p, filename = "best_robot_cross_validation.pdf", device = cairo_pdf, height = 8, width = 8)
