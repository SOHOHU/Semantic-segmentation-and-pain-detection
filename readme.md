

### Project Leadership Period and Overview  
From December 2024 to June 2025, I temporarily led the project "Deep Learning-Based Dairy Cow Pain Detection Method". We proposed a dairy cow pain detection method based on semantic segmentation and the dairy cow facial pain scale, named Pain-Score. To acquire the dairy cow pain dataset, I collected pain data (pain during calving) at a ranch from January 3 to January 18, 2025. Due to confidentiality agreements, the dataset cannot be made public.  

Unfortunately, due to time constraints, the paper for this project was not submitted to the target journal *Animal* before my graduation. After I graduated, the project was handed over to the next cohort of students to continue (the project ownership remains with Northwest A&F University). I can only provide the experiments and paper draft completed by myself, as well as the undergraduate thesis based on this project.

<p align="center">
  <img src="Picture/Pain-Score FlowLIne.png" alt="Pain-Score FlowLIne" width="60%" />
</p>

### Key Components of the Pain-Score Method  
As shown in the figure, the Pain-Score method consists of three key parts: a semantic segmentation module, a pain scoring module, and a pain detection module. To achieve fine segmentation of the dairy cow's face, we improved the DeeplabV3+ algorithm and named the new algorithm Pain-Deeplab. To realize accurate and real-time pain scoring for dairy cows, we combined the dairy cow facial pain scale with a deep learning classifier. Finally, based on the above, a simple method was used to implement dairy cow pain detection.  



### Improvements and Performance of Pain-Deeplab  
As shown in the figure, the Pain-Deeplab method enhances the performance of the DeeplabV3+ algorithm by introducing a channel attention mechanism, a feature fusion module, and a semantic parsing module. We verified the superior performance of Pain-Deeplab in relevant experiments. For complete experimental results, please refer to the draft of my paper.  

<p align="center">
  <img src="Picture/Pain-Deeplab FlowLIne.png" alt="Pain-Deeplab FlowLIne" width="60%" />
</p>


### Comparative Experiments for Classifier Selection  
To select a suitable classifier for dairy cow pain scoring, we also conducted relevant comparative experiments.  

<p align="center">
  <img src="Picture/Scoring Pipeline.png" alt="Pain-Deeplab FlowLIne" width="60%" />
</p>

### Nature and Significance of the Method  
The method of this project is a subjective approach (a concept in veterinary medicine). It has its advantages and disadvantages and can provide important references for the dairy industry.  


