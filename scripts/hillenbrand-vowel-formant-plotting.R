#!/usr/bin/env Rscript


data.dir <- '~/GitHub/hillenbrand-vowel-clustering'

hillenbrand.kmeans.formant.data <- 
  read.csv(file.path(data.dir, "hillenbrand-kmeans-formant-data.csv"), header = TRUE)

hillenbrand.gaussmm.formant.data <- 
  read.csv(file.path(data.dir, "hillenbrand-gaussmm-formant-data.csv"), header = TRUE)


ggplot(hillenbrand.kmeans.formant.data, aes(x = PC1, y = PC2, label = Vowel)) + 
  geom_text(aes(colour = factor(Label)), size = 5.5, alpha = 0.87) +
  scale_color_brewer(palette = 'Paired') + 
  theme_bw() +
  labs(title = 'K-means clustering of Hillenbrand vowel data') +
  theme(legend.position = 'none',
        panel.background = element_rect(fill = 'black',
                                       color = 'black'),
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_blank())

ggplot(hillenbrand.kmeans.formant.data, aes(x = PC1, y = PC2, label = Vowel)) + 
  geom_text(aes(colour = factor(Height)), size = 5.5, alpha = 0.87) +
  scale_color_brewer(type = 'div', palette = 'BrBG') + 
  theme_bw() +
  labs(title = 'K-means clustering of Hillenbrand vowel data') +
  theme(legend.position = 'bottom',
        panel.background = element_rect(fill = 'black',
                                        color = 'black'),
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_blank())

ggplot(hillenbrand.kmeans.formant.data, aes(x = PC1, y = PC2, label = Vowel)) + 
  geom_text(aes(colour = factor(Position)), size = 5.5, alpha = 0.87) +
  scale_color_brewer(type ='div', palette = 'RdBu') + 
  theme_bw() +
  labs(title = 'K-means clustering of Hillenbrand vowel data') +
  theme(legend.position = 'bottom',
        panel.background = element_rect(fill = 'black',
                                        color = 'black'),
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_blank())



# Gaussian mixture models -------------------------------------------------

ggplot(hillenbrand.gaussmm.formant.data, aes(x = PC1, y = PC2, label = Vowel)) + 
  geom_text(aes(colour = factor(Label)), size = 5.5, alpha = 0.87) +
  scale_color_brewer(palette = 'Paired') + 
  theme_bw() +
  labs(title = 'Gaussian Mixture Model clustering of Hillenbrand vowel data') +
  theme(legend.position = 'none',
        panel.background = element_rect(fill = 'black',
                                        color = 'black'),
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_blank())

ggplot(hillenbrand.gaussmm.formant.data, aes(x = PC1, y = PC2, label = Vowel)) + 
  geom_text(aes(colour = factor(Height)), size = 5.5, alpha = 0.87) +
  scale_color_brewer(type = 'div', palette = 'BrBG') + 
  theme_bw() +
  labs(title = 'Gaussian Mixture Model clustering of Hillenbrand vowel data') +
  theme(legend.position = 'bottom',
        panel.background = element_rect(fill = 'black',
                                        color = 'black'),
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_blank())

ggplot(hillenbrand.gaussmm.formant.data, aes(x = PC1, y = PC2, label = Vowel)) + 
  geom_text(aes(colour = factor(Position)), size = 5.5, alpha = 0.87) +
  scale_color_brewer(type ='div', palette = 'RdBu') + 
  theme_bw() +
  labs(title = 'Gaussian Mixture Model clustering of Hillenbrand vowel data') +
  theme(legend.position = 'bottom',
        panel.background = element_rect(fill = 'black',
                                        color = 'black'),
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_blank())

