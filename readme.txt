Identification model made with yolov4.
Able to identify people, sitting and falling.
The programs under the directory ideng are only reference programs and not demo presentations, 
but the parameters used are the same.
用yolov4製作的辨識模型.
能辨識人，坐，倒下。
目錄 ideng 下的只是參考程式並不是demo 的呈現，但使用參數是一樣的。


ideng is short for identifier engine. 
It was the recognition engine of the entire system at that time. 
This system works by having multiple client sides upload photos that 
need recognition to the server engine's file system.
The server-side ideng retrieves and 
processes the photos for recognition, 
then stores the processed photos into a historical file system (for 
debugging or troubleshooting purposes). 
Additionally,
 it records information about any detected falls into an SQLite database. 
A warning program then reads from the database to issue alerts.

ideng 取identifier engine 的意思。是當時整個系統的辨視引擎。
這個系統是多個client side 把須要辨識的相片放在server engine 的檔案系統。
server side 的ideng 取出相片辨識之後放進歷史檔案系統(以便除錯或找問題)，也把有跌倒的資訊寫入
sqllite db。警告程式再讀取DB發出告警。
