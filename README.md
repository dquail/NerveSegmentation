# Brachial Plexus segmentation of ultrasound by Convolutional neural network
Using Convolutional Neural Networks to automatically segment nerves from ultrasound images

## Abstract
Pain management through the use of indwelling catheters that block or mitigate pain source, is a promising alternative to narcotics, which bring on sevral unwanted side effects. Accurate identification of nerve structures in ultrasound images is a fundamental step in effectively inserting a catheter into a patient requiring pain management. In this paper, we look into using a UNET (Ronneberger, Fischer, & Brox 2015) convolutional neural network’s ability to segment the brachial plexus from an ultrasound image of a patient, and compare it with a more traditional type of convo- lutional neural network performing the same segmen-tation task.

## Background
Pain control in post surgery settings is a priority for health care providers. In addition to keeping the patient comfortable, pain management has other benefits. Pain control can help speed recovery and may reduce the risk of developing certain complications after surgery, including blood clots and pneumonia. This is because, if pain is controlled, patients will be more able to complete tasks such as walking and deep breathing exercises.

The most common treatment of pain is the administration of narcotics. But these narcotics have a significant downside. These side effects include nausea, itching, and drowsiness. 

A promising alternative to pain control is creating a nerve block. Unlike an epidural which controls pain over a large region of your body, a nerve block controls pain in a smaller region of the body, such as a limb. The nerve block is created by placing a thin catheter in the appropriate nerve region. The main advantage of these nerve blocks is that they avoid the side effects caused by narcotics, as stated above.

Creating a nerve block is done by using a needle to place a small catheter in the appropriate region. The main challenge in doing so is isolating the appropriate insertion place. Current methods involve using an ultrasound in real time to identify a nerve structure such as the brachial plexus. This requires the knowledge of a highly trained radiologist, and even then, is error prone. For these reasons, a less manual, and more accurate approach is desired. 

![alt text](WriteUp/images/UltrasoundNerve.png "Ultrasound nerve")