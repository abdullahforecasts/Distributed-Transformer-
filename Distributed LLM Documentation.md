First, we ran single device decoder and two device decoder. The embedding dimension was 32 and hidden dimension was 64\. We ran them both to check which was faster. The following was the result:  
Average: 0.474438 ms | Total calls: 4004 for single device decoder  
Average: 1.00542 ms | Total calls: 4004 for two device decoder

Single device was faster than two device because the communication overhead overtook any gains made due to parallelism as the model was simply too small.

To combat this, we changed the embedding dimension to 512 and hidden dimension to 1024\. The following was the result:  
Average: 41.4244 ms | Total calls: 4004 for two device  
Average: 47.1945 ms | Total calls: 4004 for single device

So now we begin to see gains made by our distributed architecture.

Now, we increased the number of devices to four and ran it. The following was the result:  
Average: 98.6824 ms | Total calls: 4004 for four device

Again, communication overhead between four devices overrode the gains made by parallelism. So, we increased the embedding dimension to 1024 and hidden dimension to 2048 and ran again. The following was the result:  
Average: 168.154 ms | Total calls: 4004 for four device  
Average: 189.728 ms | Total calls: 4004 for single device

Now again we see that gains have been made by our architecture.

Through our detailed analysis we can clearly see a correlation between model size and number of devices used. If the model is small, it would be more beneficial to use fewer devices, even just one depending on the situation, as any computation time saved due to parallelism is overtaken by the communication overhead between the devices. If the model is larger however, it makes more sense to use more devices as that is when parallelism truly brings benefits and the gains override the cost of communication. So, we need to carefully select the number of devices based on our model size otherwise our distributed architecture will bring more harm than good. 