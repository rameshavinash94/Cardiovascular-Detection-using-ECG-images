# Detection of Cardiovascular Diseases from ECG images


**Abstract:**

One of the most important tools for detecting cardiovascular problems is the electrocardiogram (ECG). Until recently, the vast majority of ECG records were kept on paper. Manually examining ECG paper records can be a difficult and time-consuming process. 

If we digitize such paper ECG records, we can perform automated diagnosis and analysis. The goal of this project is to use image processing and machine learning techniques to convert ECG images into a 1-D signal and extract P, QRS, and T waves that exist in ECG signals to demonstrate the electrical activity of the heart using various techniques. Post feature extraction it can aid in the diagnosis of most cardiac diseases.

**Datasets:**

ECG images: https://data.mendeley.com/datasets/gwbz3fsgp8/2 

The above dataset contains ECG image signals from both healthy individuals and persons with cardiovascular problems.

**PPT LINK:**

https://www.canva.com/design/DAEx1Se7iPs/Xy1tWQVcBt3Oww8hBHEvRg/view?utm_content=DAEx1Se7iPs&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink

**DEPLOYMENT LINK:**

We have deployed our applicaiton in GCP.

https://cmpe255-project-q4uake3apq-uc.a.run.app

Download any image from the below folder path and try uploading it to the above url to get real-time insights.
https://github.com/rameshavinash94/Cardiovascular-Detection-using-ECG-images/tree/main/ECG_IMAGES_DATASET

**Approach:**

The user uploads an ECG image to our web app. Then, we use techniques like rgb2gray conversion, gaussian filtering, resizing, and thresholding to extract only the signals that do not have grid lines. The required waves (P, QRS, T) are then extracted using contour techniques and converted to a 1D signal. The normalized 1D signal is then fed into our pre-trained ML model, which is then analyzed. When the model has completed the analysis, it returns the results to the user based on the findings.

Here, we have used 4 categories for image classification for our ECG images.
_Normal
Myocardial infarction
Abnormal Heart beat
History of Myocardial infarction_

One benefit of our app is that the user can view the entire workflow in the UI and receive real-time feedback.

The tricky path here is feature extraction from images; if done correctly and optimally, the accuracy of our model can be increased.

**Abstract Link**: https://github.com/rameshavinash94/Cardiovascular-Detection-using-ECG-images/blob/main/docs/Digitizing%20ECG%20signals%20and%20detection%20of%20Cardiovascular%20diseases.docx.pdf

**PROJECT REPORT:** https://github.com/rameshavinash94/Cardiovascular-Detection-using-ECG-images/blob/main/docs/Final%20project%20submission.pdf


![](https://raw.githubusercontent.com/rameshavinash94/Cardiovascular-Detection-using-ECG-images/main/img/Architecture_Diagram.png)

![](https://raw.githubusercontent.com/rameshavinash94/Cardiovascular-Detection-using-ECG-images/main/img/Deployment_diagram.png)

**SAMPLE DEMO GIF**

![](https://raw.githubusercontent.com/rameshavinash94/Cardiovascular-Detection-using-ECG-images/main/img/demo.gif)


