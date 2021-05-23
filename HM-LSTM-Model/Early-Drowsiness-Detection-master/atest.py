st=0
            st2=0
            if (First_frame == False):
                leftEye=leftEye.astype(np.float32)
                rightEye = rightEye.astype(np.float32)
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray,leftEye, None, **lk_params)
                p2, st2, err2 = cv2.calcOpticalFlowPyrLK(old_gray, gray, rightEye, None, **lk_params)

            if np.sum(st)+np.sum(st2)==12 and First_frame==False:

                p1 = np.round(p1).astype(np.int)
                p2 = np.round(p2).astype(np.int)
                #print(p1)

                leftEAR = eye_aspect_ratio(p1)
                rightEAR = eye_aspect_ratio(p2)

                ear = (leftEAR + rightEAR) / 2.0
                EAR_series = shift(EAR_series, -1, cval=ear)
                leftEyeHull = cv2.convexHull(p1)
                rightEyeHull = cv2.convexHull(p2)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                old_gray = gray.copy()
                leftEye = p1
                rightEye = p2
                ############HANDLING THE EMERGENCY SITATION################
                ###########################################################
                ###########################################################
                COUNTER = EMERGENCY(ear, COUNTER)

            if Q.full() and (reference_frame>15):
                IF_Closed_Eyes = loaded_svm.predict(EAR_series.reshape(1,-1))
                if Counter4blinks==0:
                    Current_Blink = Blink()
                    retrieved_blinks, TOTAL_BLINKS, Counter4blinks, BLINK_READY, skip = Blink_Tracker(EAR_series[6],
                                                                                                      IF_Closed_Eyes,
                                                                                                      Counter4blinks,
                                                                                                      TOTAL_BLINKS, skip)
                if (BLINK_READY==True):
                    reference_frame=20   #initialize to a random number to avoid overflow in large numbers
                    skip = True
                    #####
                    BLINK_FRAME_FREQ = TOTAL_BLINKS / number_of_frames
                    for detected_blink in retrieved_blinks:
                        print(detected_blink.amplitude, Last_Blink.amplitude)
                        print(detected_blink.duration, Last_Blink.duration)
                        print('-------------------')
                        with open(output_file, 'ab') as f_handle:
                            f_handle.write(b'\n')
                            np.savetxt(f_handle,[TOTAL_BLINKS,BLINK_FRAME_FREQ*100,detected_blink.amplitude,detected_blink.duration,detected_blink.velocity], delimiter=', ', newline=' ',fmt='%.4f')

                    Last_Blink.end = -10 # re initialization


                    #####

                line.set_ydata(EAR_series)
                plot_frame.draw()
                frameMinus7=Q.get()
                cv2.imshow("Frame", frameMinus7)
            elif Q.full():
                junk = Q.get()

            key = cv2.waitKey(1) & 0xFF


            if key != 0xFF:
                 break