This website contains results, code and pre-trained models from the paper [Sequence-to-sequence modeling for action identification at high temporal resolution](https://arxiv.org/abs/2111.02521) by Aakash Kaku\*, Kangning Liu\*, Avinash Parnandi\*, Haresh Rengaraj Rajamohan, Kannan Venkataramanan, Anita Venkatesan, Audre Wirtanen, Natasha Pandit, Heidi Schambra^, Carlos Fernandez-Granda^ [\* - Equal Contribution, ^ - Joint last authors].

## Quantification of Stroke Rehabilitation
- Stroke is a leading cause of motor impairment and the numbers for stroke are increasing.
<p align="center">
  <img src="https://user-images.githubusercontent.com/32464452/145431460-e4154d68-7c80-4ef3-91aa-35faf45ea5b5.png" />
</p>
- After stroke, there is some spontaneous recovery that occurs because of brain plasticity, but recovery is often incomplete. However, it is believed that if we intervene early after stroke, therapies like rehabilitation training could improve the recovery.
<p align="center">
  <img src="https://user-images.githubusercontent.com/32464452/145431790-366a13c0-9ac7-454d-81b4-49e62c8ac01c.png" />
</p>
- Rehabilitation training involves repeatedly practicing the activities of daily living -- ADLs. The ADLs are composed of 5 fundamental motions: __reach__, __transport__, __reposition__, __stabilize__ and __idle__. For example, if we want to drink water from a glass, we would start from __idle__, __reach__ for the glass, __transport__ the glass towards our mouth, __transport__ the glass back to the table, and __reposition__ the arms in the starting position.
- A major question is what is the optimal dose of rehab training. In animal studies, we have quantified the doses. For humans, some studies suggest that we might be under-dosing our patients by a factor of **ten**. But, **the optimal dose of training is unknown**.
- Currently, the best way to quantify rehabilitation is hand tallying. It takes **one hour** of manual effort to label **one minute** of recorded training. This approach is slow, expensive, and not scalable.
- Therefore, we attempt to solve this problem by using machine learning to identify fundamental motions automatically from wearable-sensor data in real time. Here, we can see an individual performing an activity. We capture his motion using nine - sensors attached to his upper body. The sensors capture various signals like joint angles, accelerations, and quaternions. Human coders, then, use synchronously recorded video to label the fundamental motions.
<p align="center">
  <img src="https://user-images.githubusercontent.com/32464452/144506546-72d62b1f-7ef2-4cc0-8805-9d6d34aa09cb.gif" />
</p>

## Seq2seq model predicting extremely fine-grained actions
<p align="center">
  <img src="https://user-images.githubusercontent.com/32464452/144508990-195293f4-311b-469d-a2cd-92ff2841122e.png" />
</p>
Comparison of sequence-to-sequence (seq2seq) and segmentation models. The segmentation model outputs frame-wise action predictions, which can then be converted to a sequence
estimate by removing the duplicates. The seq2seq model produces a sequence estimate directly.

## Segmentation models cannot detect boundaries for extremely fine-grained actions
<p align="center">
  <img src="https://user-images.githubusercontent.com/32464452/144508026-c03afa71-b454-484d-bddd-7f990372858e.png" />
</p>
Boundary accuracy achieved by the segmentation models vs duration of the actions for several datasets. Boundary-detection accuracy is inversely proportional to action duration.

## Performance Metric
In order to evaluate sequence predictions we use two metrics based on the Levenshtein distance: edit score (ES) and action error rate (AER) (inspired by the word-error rate metric used in speech recognition). The Levenshtein distance, L(G, P), is the minimum number of insertions, deletions, and substitutions required to convert a predicted sequence P to a ground-truth sequence G. For example, if G = [reach, idle, stabilize] and P = [reach, transport], then L(G, P) = 2 (transport is substituted for idle and stabilize is inserted). We have:
![image](https://user-images.githubusercontent.com/32464452/144508527-d6b8084a-0f45-46d4-aa0e-5e972ba18712.png)
where len(G) and len(P) are the lengths of the ground-truth and predicted sequence respectively. The edit score is more lenient when the estimated sequence is longer. In contrast, AER penalizes longer and shorter predictions equally. For example, if G = [reach, idle, stabilize], P1 = [reach,idle], and P2 = [reach, idle, stabilize, transport], then ES(G, P1) = 0.67 and ES(G, P2) = 0.75, but AER(G, P1) = AER(G, P2) = 0.33.

## Results
- **StrokeRehab dataset**
![image](https://user-images.githubusercontent.com/32464452/144508233-17f6920b-2c1a-44d0-a5ec-a1bfe1192bd2.png)
Results on StrokeRehab: Seq2seq outperforms segmentation-based approaches. We report mean (95% confidence interval) which is computed via bootstrapping.

- **Action-recognition benchmarks datasets**
![image](https://user-images.githubusercontent.com/32464452/144508275-282b8ede-9f09-4c8d-b72e-035984417f01.png)
Results on action-recognition benchmarks: Seg2seq, the seq2seq model which uses the output of a pretrained segmentation-based model, outperforms segmentation-based approaches.

- **Count of primitives for StrokeRehab dataset**
In stroke rehabilitation, action identification can be used for quantifying dose by counting functional primitives. The figure below shows that the raw2seq version of the seq2seq model produces accurate counts for all activities in the StrokeRehab dataset. Performance is particularly good for structured activities such as moving objects on/off a shelf, in comparison to less structured activities such as brushing, which tend to be more heterogeneous across patients
<p align="center">
  <img src="https://user-images.githubusercontent.com/32464452/144508718-6b122fe9-2fe8-4a47-9142-14733c6cd923.png" />
</p>
Comparison of ground-truth and predicted mean counts for the different activities in the StrokeRehab dataset. The relative error is very small for structured activities like moving objects on/off a shelf (Shelf), and larger for unstructured activities like brushing.


## Pre-Trained Models and Code
Please visit [our github page](https://github.com/aakashrkaku/seq2seq_hrar) for data, pre-trained models, code and instructions on how to use the code. 

