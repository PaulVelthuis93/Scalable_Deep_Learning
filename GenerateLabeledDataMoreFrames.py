# -*- coding: utf-8 -*-
import csv
import cv2
import os
#adapted from github: https://github.com/dllatas/deepLearning/blob/master/helper/generate_frame_from_video.py
"""
1 (e) Angry
- AU 4+5+15+17,
2 (f) Contempt - AU 14,
3 (a) Disgust - AU 1+4+15+17,
4 (d) Fear
- AU 1+4+7+20,
5 (b) Happy
- AU 6+12+25,
6 (g) Sadness - AU 1+2+4+15+17
7 (c) Surprise - AU 1+2+5+25+27
SURPRISE AND FEAR GG
"""

def change_to_video_name(csv_name, suffix):
    return csv_name[:-10]+"."+suffix

def generate_frame(video_path, video_name, second, label, dest_path, td):
    print "video_path", video_path
    print 'video_name',video_name
    print 'second',second
    print 'label',label
    print 'dest_path',dest_path

    if td !=-1:
        #take images in time window
        print td
        #loop through possibilities 14 frames a second, so if td = 1/14 of a second take 1 td 2/14 second take 2 etcetera
        second_new = second
        time=second+td
        print time
        while second_new <= time:
            vidcap = cv2.VideoCapture(os.path.join(video_path, video_name))
            vidcap.set(0, int(second_new*1000))
            success, image = vidcap.read()            
            if success:         
                
                cv2.imwrite(os.path.join(dest_path, video_name+"_"+str((second_new - (1/14)))+"_"+str(label)+".jpg"), image)
                #1/4 doesn't work hard value works
                second_new = second_new + 0.0714
                print 'second new', second_new
            second_new = second_new + 0.0714

    else:
        vidcap = cv2.VideoCapture(os.path.join(video_path, video_name))
        vidcap.set(0, int(second*1000))
        success, image = vidcap.read()
        if success:
            cv2.imwrite(os.path.join(dest_path, video_name+"_"+str(second)+"_"+str(label)+".jpg"), image)

def check_angry(content):
    baseline = 50
    #disgust = ["AU4", "AU15", "AU17"]
    sadness = ["AU2", "AU4", "AU15", "AU17"]
    #angry = ["AU4", "AU5", "AU15", "AU17"]
    label = 1 # 159

    # print 'content:',content
    emotion_time = content[0][1]
    emotion = []
    for c in content:
        for h in sadness:
            if c[0] == h:
                emotion.append(c[1])
    print emotion
    factor = sum(emotion)/len(sadness)
    if factor >= baseline:
        return emotion_time, label

def check_contempt(content):
    baseline = 100
    contempt = ["AU14"]
    label = 2
    emotion_time = content[0][1]
    for c in content:
        for h in contempt:
            if c[0] == h and c[1] >= baseline:
                return emotion_time, label
"""
Use this function to check for which time a person is happy and store the time and happy label
"""
def check_happiness(content):
    baseline = 100
    happiness = ["Smile"]
    label = 5
    # print content
    emotion_time = content[0][1]
    # print 'emotion_time',emotion_time
    for c in content:
        for h in happiness:
            # print h
            if c[0] == h and c[1] >= baseline:
                print 'emotion & label',emotion_time, label
                return emotion_time, label
"""
Use this function to check for which time a person is happy and store the time and happy label
"""
def check_happiness(content):
    baseline = 50
    happiness = ["Smile"]
    label = 5
    # print content
    emotion_time = content[0][1]
    # print 'emotion_time',emotion_time
    for c in content:
        for h in happiness:
            # print h
            if c[0] == h and c[1] >= baseline:
                print 'emotion & label',emotion_time, label
                return emotion_time, label

def check_nonHappiness(content):
    baseline = 0
    happiness = ["Smile"]
    AU12baseline = 20
    label = 50
    # print content
    emotion_time = content[0][1]
    # print 'emotion_time',emotion_time
    for c in content:
        for h in happiness:
            # print h
            # the hapiness value is exactly 0 and AU12 is below AU12 baseline, then a person is non happy
            if c[0] == h and c[1] == baseline and content[12][1] <= AU12baseline and content[13][1] <= AU12baseline:
                print 'emotion & label',emotion_time, label
                return emotion_time, label

def get_content(header, row):
    """
    return: time frames for each AU in video
    """
    # print 'row',row
    content = row[0:]
    result = []
    for h in header:
        # print 'ts',h
        result.append([h[0], float(content[h[1]])])
    # print result
    return result

def get_header_au(row):
    rules = ["Time", "Smile", "AU"]
    #header = row[0:2]
    header=row
    #print row
    result = []
    i = 0
    #for all values in the header
    for h in header:
        print h
        if h in rules or h[0:2] in rules or 'AU' in h:
            result.append([h, i])
        i = i + 1
    # print result
    return result

def process_video_happiness(csv_path, video_path, dest_path, suffix):
    for root, dirs, files in os.walk(csv_path, True):
        for name in files:
            with open(os.path.join(root, name), 'rU') as csvfile:
                reader = csv.reader((line.replace('\0','') for line in csvfile), delimiter=',', quotechar='|')
                t2 = 0
                for row in reader:
                    # print row
                    if reader.line_num == 1:
                        header = get_header_au(row)
                    else:
                        # print 'h',header
                        content = get_content(header, row)
                        # if len(header) > 0:
                        if len(header) > 0:
                            if content:
                                content = get_content(header, row)
                                # emotion = check_angry(content)
                                emotion = check_happiness(content)
                                emotion_time1 = content[0][1]
                                print 'emotion_time1',emotion_time1
                                #get timewindow
                                td = -1
                                try:
                                    if reader.next():
                                        content2=get_content(header, reader.next())
                                        emotion_time2 = content2[0][1]
                                        td = emotion_time2 - emotion_time1
                                        print 'td',td
                                except:
                                    pass
                                if emotion is not None:
                                    # print emotion[0]
                                    # print emotion[1]
                                    generate_frame(video_path, change_to_video_name(name, suffix), emotion[0], emotion[1], dest_path, td)
                                        
def process_video_non_happiness(csv_path, video_path, dest_path, suffix):
    for root, dirs, files in os.walk(csv_path, True):
        for name in files:
            with open(os.path.join(root, name), 'rU') as csvfile:
                reader = csv.reader((line.replace('\0','') for line in csvfile), delimiter=',', quotechar='|')
                for row in reader:
                    # print row
                    if reader.line_num == 1:
                        header = get_header_au(row)
                    
                    else:
                        # print 'h',header
                        content = get_content(header, row)
                        # if len(header) > 0:

                        if len(header) > 0:
                            if content:
                                content = get_content(header, row)
                                #emotion = check_angry(content)
                                # print emotion
                                emotion = check_nonHappiness(content)
                                #emotion = check_happiness(content)
                                emotion_time1 = content[0][1]
                                print 'emotion_time1',emotion_time1
                                #get timewindow
                                td = -1
                                try:
                                    if reader.next():
                                        content2=get_content(header, reader.next())
                                        emotion_time2 = content2[0][1]
                                        td = emotion_time2 - emotion_time1
                                        print 'td',td
                                except:
                                    pass
                                if emotion is not None:
                                    print emotion[0]
                                    print emotion[1]
                                    generate_frame(video_path,
                                        change_to_video_name(name, suffix), emotion[0], emotion[1], dest_path,td) 
                                        


                               

def main(argv=None): # pylint: disable=unused-argument
    csv_path = "AMFED/AMFED/AU_Labels"
    video_path = "AMFED/AMFED/Videos_FLV"
    #destination path for hapiness
    #dest_path = "AMFED/AMFED/happiness"
    suffix = "flv"
    #process_video_hapiness(csv_path, video_path, dest_path, suffix)
    #destination path for sadness
    #dest_path = "AMFED/AMFED/sadness"
    dest_path = "AMFED/AMFED/nonHappinessFourteenFramesPerSecond"
    process_video_non_happiness(csv_path, video_path, dest_path, suffix)

if __name__ == '__main__':
    main()
