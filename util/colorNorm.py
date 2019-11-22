#coding:utf-8
from PIL import Image, ImageStat
import openslide
import argparse
import os, re, shutil, sys, time
import numpy as np
import cv2
import random
import math

#parser = argparse.ArgumentParser(description='Divide *.svs file into block images.')
#parser.add_argument('fn', help='*.svs file\'s path')
#args = parser.parse_args()

b_size = 224    #ブロック画像のサイズ
t_size = 16 #サムネイル画像中の1ブロックのサイズ
ext = '.tif'    #出力画像の拡張子

def DivideSVS(img, w_num, h_num, b_size):   #b_size x b_sizeのブロック画像を同名のディレクトリに出力
    #切り出す幅と高さ
    print('Width:'+str(w_num))
    print('Height:'+str(h_num))
    #縦の分割枚数
    for h1 in range(h_num):
        #横の分割枚数
        for w1 in range(w_num):
            w2 = w1 * b_size
            h2 = h1 * b_size
            sys.stdout.flush()
            sys.stdout.write('\r'+str(w1+h1*w_num+1)+' / '+str(w_num*h_num))
            yield img.read_region((w2,h2),0,(b_size,b_size))

def MakeThumb(dir, w_num, h_num, t_size):   #サムネイルと彩度分布を生成
    thumb = Image.new('RGB',(w_num * t_size, h_num * t_size))   #標本サムネイル
    #thumb_s = Image.new('L',(w_num, h_num)) #彩度分布画像
    for h1 in range(h_num):
        for w1 in range(w_num):
            b_img = Image.open('./patch/'+dir+'/'+dir+'_'+str(w1 + h1 * w_num)+ext) #ブロック画像読み込み
            r_img = b_img.resize((t_size, t_size), Image.BILINEAR)  #サムネイル用に縮小
            thumb.paste(r_img, (w1 * t_size, h1 * t_size))  #対応する箇所に縮小画像を貼り付け
            #h_img, s_img, v_img = b_img.convert('HSV').split()  #RGB→HSV変換，チャンネルを分離
            #stat = ImageStat.Stat(h_img)    #彩度画像の統計量取得
            #thumb_s.putpixel((w1,h1),round(stat.mean[0]))   #彩度のブロック平均値を書き込み
    thumb.save('./patch/'+dir+'/'+dir+'_thumb'+ext)    #標本サムネイル保存
    #thumb_s.save('./patch/'+dir+'/'+dir+'_hue_m'+ext)    #彩度分布画像保存

def SampleImage(dir):   #ディレクトリ内のガラス領域の画像を削除（or移動）
    f = open('./'+dir+'/i_num.txt', 'r')    #画像枚数読み込み
    i_num = []
    line = f.readline()
    while line:
        i_num.append(int(re.sub(r'\D', '', line)))   #数字部分を抽出
        line = f.readline()
    thumb_s = Image.open(dir+'/'+dir+'_sat'+ext)    #彩度分布画像読み込み
    array_s = np.asarray(thumb_s)   #cv形式に変換
    ret, array_bi = cv2.threshold(array_s, 0, 255, cv2.THRESH_OTSU) #判別分析法で二値化
    thumb_bi = Image.fromarray(array_bi)    #PIL形式に変換
    thumb_bi.save(dir+'/'+dir+'_mask'+ext)    #抽出領域確認用
    for h1 in range(i_num[1]):
        for w1 in range(i_num[0]):
            if thumb_bi.getpixel((w1,h1)) == 0:
                #ファイル削除
                os.remove(dir+'/'+dir+'_'+str(w1+h1*i_num[0])+ext)
                #別のフォルダに移動（確認用）
                #if not os.path.exists(dir+'/white'):
                #    os.mkdir(dir+'/white')
                #shutil.move(dir+'/'+dir+'_'+str(w1+h1*i_num[0])+ext,dir+'/white')

def RandomSampling(dir, i_num):   #ディレクトリ内の画像を任意の数ランダムサンプリング
    if not os.path.exists(dir+'_rs'):
        os.mkdir(dir+'_rs')
    files = os.listdir(dir)
    files.remove(dir+'_mask'+ext)
    files.remove(dir+'_sat'+ext)
    files.remove(dir+'_thumb'+ext)
    files.remove('i_num.txt')
    if len(files) <= i_num:
        for i in range(len(files)):
            shutil.move(dir+'/'+files[i],dir+'_rs')
    else:
        for i in range(i_num):
            tmp = files.pop(random.randint(0,len(files)-1))
            shutil.move(dir+'/'+tmp,dir+'_rs')

def ROIdetection(src):
    #画像読み込み
    thumb_s = np.array(Image.open('./patch/'+src+'/'+src+'_thumb'+ext))
    #チャンネル分割，彩度マスク作成
    R_s, G_s, B_s = cv2.split(thumb_s)
    Max_s = np.maximum(np.maximum(R_s, G_s), B_s)
    Min_s = np.minimum(np.minimum(R_s, G_s), B_s)
    Sat_s = Max_s - Min_s
    ret_s, mask_s = cv2.threshold(Sat_s, 0, 1, cv2.THRESH_OTSU) #初期組織領域抽出
    #白色信号値算出（各チャンネル最大値）
    R_max_s = R_s.max()
    G_max_s = G_s.max()
    B_max_s = B_s.max()
    #各画素のOpticalDensity計算
    D_Rs = - np.log(np.clip((R_s + 0.1)/(R_max_s + 0.1), 0.0, 1.0))
    D_Gs = - np.log(np.clip((G_s + 0.1)/(G_max_s + 0.1), 0.0, 1.0))
    D_Bs = - np.log(np.clip((B_s + 0.1)/(B_max_s + 0.1), 0.0, 1.0))
    D_s = (D_Rs + D_Gs + D_Bs) / 3
    #非関心領域の決定
    RBC = D_Bs - D_Rs #吸光度の差分計算
    RBC_mean_s, RBC_sd_s = cv2.meanStdDev(RBC, mask_s)
    #彩度マスク
    Sat_i = Image.fromarray(Sat_s)
    Sat_ir = Sat_i.resize((int(Sat_i.width/t_size), int(Sat_i.height/t_size)), Image.BILINEAR)  #サムネイル用に縮小
    Sat_ar = np.array(Sat_ir)
    ret_sr, mask_sr = cv2.threshold(Sat_ar, 0, 1, cv2.THRESH_OTSU) #組織領域抽出
    kernel = np.ones((3,3),np.uint8)
    mask_tmp = cv2.erode(mask_sr, kernel, iterations = 1) #モルフォロジー処理
    if mask_tmp.sum() > 100:    #100画素以上なら採用
        mask_sr = mask_tmp
    #RBCマスク
    mask_r = np.where(RBC > RBC_mean_s + 2 * RBC_sd_s, 0, mask_s) #非関心領域の決定
    mask_i = Image.fromarray(mask_r)
    mask_ir = mask_i.resize((int(mask_i.width/t_size), int(mask_i.height/t_size)), Image.BILINEAR)  #サムネイル用に縮小
    mask_ar = np.array(mask_ir)
    ret_rr, mask_rr = cv2.threshold(mask_ar, 0, 1, cv2.THRESH_OTSU) #組織領域抽出
    mask_rr = np.where(mask_rr == 1, mask_sr, 0) #非関心領域の決定
    #最終マスク出力
    if mask_rr.sum() < 100:
        thumb_o = Image.fromarray(np.uint8(np.clip(mask_s * 255, 0, 255)))
        thumb_or = Image.fromarray(np.uint8(np.clip(mask_sr * 255, 0, 255)))
    else:
        thumb_o = Image.fromarray(np.uint8(np.clip(mask_s * 255, 0, 255)))
        thumb_or = Image.fromarray(np.uint8(np.clip(mask_rr * 255, 0, 255)))
    thumb_o.save('./patch/'+src+'/'+src+'_mask'+ext)
    thumb_or.save('./patch/'+src+'/'+src+'_mask_r'+ext)

def ColorNorm_patch(src, tar):
    #画像読み込み
    thumb_s = np.array(Image.open('./patch/'+src+'/'+src+'_thumb'+ext)) #ソース画像サムネイル
    thumb_t = np.array(Image.open('./patch/'+tar+'/'+tar+'_thumb'+ext)) #ターゲット画像サムネイル
    mask_s = np.array(Image.open('./patch/'+src+'/'+src+'_mask'+ext)) #ソース画像マスク
    mask_t = np.array(Image.open('./patch/'+tar+'/'+tar+'_mask'+ext)) #ターゲット画像マスク
    mask_r = np.array(Image.open('./patch/'+src+'/'+src+'_mask_r'+ext)) #処理パッチマスク
    if not os.path.exists('./patch_cn/'+src+'_cn'):
        os.mkdir('./patch_cn/'+src+'_cn') #変換後のディレクトリ作成
    #チャンネル分割
    R_s, G_s, B_s = cv2.split(thumb_s)
    R_t, G_t, B_t = cv2.split(thumb_t)
    #白色信号値算出
    R_max_s = R_s.max()
    G_max_s = G_s.max()
    B_max_s = B_s.max()
    R_max_t = R_t.max()
    G_max_t = G_t.max()
    B_max_t = B_t.max()
    #各画素のOpticalDensity計算
    D_Rs = - np.log(np.clip((R_s+0.01)/(R_max_s+0.01), 0.0, 1.0))
    D_Gs = - np.log(np.clip((G_s+0.01)/(G_max_s+0.01), 0.0, 1.0))
    D_Bs = - np.log(np.clip((B_s+0.01)/(B_max_s+0.01), 0.0, 1.0))
    D_Rt = - np.log(np.clip((R_t+0.01)/(R_max_t+0.01), 0.0, 1.0))
    D_Gt = - np.log(np.clip((G_t+0.01)/(G_max_t+0.01), 0.0, 1.0))
    D_Bt = - np.log(np.clip((B_t+0.01)/(B_max_t+0.01), 0.0, 1.0))
    D_s = (D_Rs + D_Gs + D_Bs) / 3
    D_t = (D_Rt + D_Gt + D_Bt) / 3
    #cx, cyを計算
    cx_s = np.where(D_s == 0, 0, D_Rs / D_s - 1)
    cy_s = np.where(D_s == 0, 0, (D_Gs - D_Bs) / (math.sqrt(3) * D_s))
    cx_t = np.where(D_t == 0, 0, D_Rt / D_t - 1)
    cy_t = np.where(D_t == 0, 0, (D_Gt - D_Bt) / (math.sqrt(3) * D_t))
    #各要素の平均と標準偏差を計算
    D_mean_s, D_sd_s = cv2.meanStdDev(D_s, mask_s)
    cx_mean_s, cx_sd_s = cv2.meanStdDev(cx_s, mask_s)
    cy_mean_s, cy_sd_s = cv2.meanStdDev(cy_s, mask_s)
    D_mean_t, D_sd_t = cv2.meanStdDev(D_t, mask_t)
    cx_mean_t, cx_sd_t = cv2.meanStdDev(cx_t, mask_t)
    cy_mean_t, cy_sd_t = cv2.meanStdDev(cy_t, mask_t)
    RBC = D_Bs - D_Rs #吸光度の差分計算（外れ値となる赤血球らしさ）
    RBC_mean_s, RBC_sd_s = cv2.meanStdDev(RBC, mask_s)
    #変換先の色分布へ補正
    for h in range(mask_r.shape[0]):
        for w in range(mask_r.shape[1]):
            if mask_r[h,w] == 255 and os.path.exists('./patch/'+src+'/'+src+'_'+str(h*mask_r.shape[1]+w)+ext):
                img_i = np.array(Image.open('./patch/'+src+'/'+src+'_'+str(h*mask_r.shape[1]+w)+ext))
                R_i, G_i, B_i, A_i = cv2.split(img_i)
                #各画素のOpticalDensity計算
                D_Ri = - np.log(np.clip((R_i+0.01)/(R_max_s+0.01), 0.0, 1.0))
                D_Gi = - np.log(np.clip((G_i+0.01)/(G_max_s+0.01), 0.0, 1.0))
                D_Bi = - np.log(np.clip((B_i+0.01)/(B_max_s+0.01), 0.0, 1.0))
                D_i = (D_Ri + D_Gi + D_Bi) / 3
                #cx, cyを計算
                cx_i = np.where(D_i == 0, 0, D_Ri / D_i - 1)
                cy_i = np.where(D_i == 0, 0, (D_Gi - D_Bi) / (math.sqrt(3) * D_i))
                #変換先に標準化
                #D_c = (D_i - D_mean_s[0]) * D_sd_t[0] / D_sd_s[0] + D_mean_t[0]
                D_c = D_i
                cx_c = (cx_i - cx_mean_s[0]) * cx_sd_t[0] / cx_sd_s[0] + cx_mean_t[0]
                cy_c = (cy_i - cy_mean_s[0]) * cy_sd_t[0] / cy_sd_s[0] + cy_mean_t[0]
                weight = np.square(np.clip(((D_Bi - D_Ri) - RBC_mean_s) / (2 * RBC_sd_s), 0, 1)) #平均から標準x2まで線形な重み
                #weight = np.clip(((D_Bi - D_Ri) - RBC_mean_s) / (3 * RBC_sd_s), 0, 1)
                #D_c = np.where((D_Bi - D_Ri) > RBC_mean_s + 2 * RBC_sd_s, D_i, D_i) #非関心領域の決定
                #cx_c = np.where((D_Bi - D_Ri) > RBC_mean_s + 2 * RBC_sd_s, cx_i, cx_c)
                #cy_c = np.where((D_Bi - D_Ri) > RBC_mean_s + 2 * RBC_sd_s, cy_i, cy_c)
                #D_c = weight * D_i + (1 - weight) * D_c
                D_c = D_i
                cx_c = weight * cx_i + (1 - weight) * cx_c
                cy_c = weight * cy_i + (1 - weight) * cy_c
                D_Rc = D_c * (cx_c + 1)
                D_Gc = D_c * (2 - cx_c + math.sqrt(3) * cy_c) / 2
                D_Bc = D_c * (2 - cx_c - math.sqrt(3) * cy_c) / 2
                R_c = np.exp(-D_Rc) * R_max_t
                G_c = np.exp(-D_Gc) * G_max_t
                B_c = np.exp(-D_Bc) * B_max_t
                R_cvt = Image.fromarray(np.uint8(np.clip(R_c, 0.0, 255.0)))
                G_cvt = Image.fromarray(np.uint8(np.clip(G_c, 0.0, 255.0)))
                B_cvt = Image.fromarray(np.uint8(np.clip(B_c, 0.0, 255.0)))
                img_cvt = Image.merge('RGB', (R_cvt,G_cvt,B_cvt))
                img_cvt.save('./patch_cn/'+src+'_cn/'+src+'_'+str(h*mask_r.shape[1]+w)+ext)
            #else:
                #os.remove('./patch/'+src+'/'+src+'_'+str(h*mask_r.shape[1]+w)+ext)
    #変換後サムネイル作成
    weight = np.square(np.clip(((D_Bs - D_Rs) - RBC_mean_s) / (2 * RBC_sd_s), 0, 1))
    #weight = np.clip(((D_Bs - D_Rs) - RBC_mean_s) / (3 * RBC_sd_s), 0, 1)
    weight = np.where(mask_s == 255, weight, 1)
    #D_c = (D_s - D_mean_s[0]) * D_sd_t[0] / D_sd_s[0] + D_mean_t[0]
    D_c = D_s
    cx_c = (cx_s - cx_mean_s[0]) * cx_sd_t[0] / cx_sd_s[0] + cx_mean_t[0]
    cy_c = (cy_s - cy_mean_s[0]) * cy_sd_t[0] / cy_sd_s[0] + cy_mean_t[0]
    D_c = weight * D_s + (1 - weight) * D_c
    cx_c = weight * cx_s + (1 - weight) * cx_c
    cy_c = weight * cy_s + (1 - weight) * cy_c
    D_Rc = D_c * (cx_c + 1)
    D_Gc = D_c * (2 - cx_c + math.sqrt(3) * cy_c) / 2
    D_Bc = D_c * (2 - cx_c - math.sqrt(3) * cy_c) / 2
    R_c = np.exp(-D_Rc) * R_max_t
    G_c = np.exp(-D_Gc) * G_max_t
    B_c = np.exp(-D_Bc) * B_max_t
    R_cvt = Image.fromarray(np.uint8(np.clip(R_c, 0.0, 255.0)))
    G_cvt = Image.fromarray(np.uint8(np.clip(G_c, 0.0, 255.0)))
    B_cvt = Image.fromarray(np.uint8(np.clip(B_c, 0.0, 255.0)))
    img_cvt = Image.merge('RGB', (R_cvt,G_cvt,B_cvt))
    img_cvt.save('./patch_cn/'+src+'_cn/'+src+'_thumb'+ext)

def ColorNormThumb(src, tar):
    #画像読み込み
    thumb_s = np.array(Image.open('./thumb/'+src))
    thumb_t = np.array(Image.open('./thumb/'+tar))
    mask_s = np.array(Image.open('./mask/'+src))
    mask_t = np.array(Image.open('./mask/'+tar))
    mask_r = np.array(Image.open('./mask_r/'+src))
    #チャンネル分割
    R_s, G_s, B_s = cv2.split(thumb_s)
    R_t, G_t, B_t = cv2.split(thumb_t)
    #白色信号値算出
    R_max_s = R_s.max()
    G_max_s = G_s.max()
    B_max_s = B_s.max()
    R_max_t = R_t.max()
    G_max_t = G_t.max()
    B_max_t = B_t.max()
    #各画素のOpticalDensity計算
    D_Rs = - np.log(np.clip((R_s+0.01)/(R_max_s+0.01), 0.0, 1.0))
    D_Gs = - np.log(np.clip((G_s+0.01)/(G_max_s+0.01), 0.0, 1.0))
    D_Bs = - np.log(np.clip((B_s+0.01)/(B_max_s+0.01), 0.0, 1.0))
    D_Rt = - np.log(np.clip((R_t+0.01)/(R_max_t+0.01), 0.0, 1.0))
    D_Gt = - np.log(np.clip((G_t+0.01)/(G_max_t+0.01), 0.0, 1.0))
    D_Bt = - np.log(np.clip((B_t+0.01)/(B_max_t+0.01), 0.0, 1.0))
    D_s = (D_Rs + D_Gs + D_Bs) / 3
    D_t = (D_Rt + D_Gt + D_Bt) / 3
    #cx, cyを計算
    cx_s = np.where(D_s == 0, 0, D_Rs / D_s - 1)
    cy_s = np.where(D_s == 0, 0, (D_Gs - D_Bs) / (math.sqrt(3) * D_s))
    cx_t = np.where(D_t == 0, 0, D_Rt / D_t - 1)
    cy_t = np.where(D_t == 0, 0, (D_Gt - D_Bt) / (math.sqrt(3) * D_t))
    #各要素の平均と標準偏差を計算
    D_mean_s, D_sd_s = cv2.meanStdDev(D_s, mask_s)
    cx_mean_s, cx_sd_s = cv2.meanStdDev(cx_s, mask_s)
    cy_mean_s, cy_sd_s = cv2.meanStdDev(cy_s, mask_s)
    D_mean_t, D_sd_t = cv2.meanStdDev(D_t, mask_t)
    cx_mean_t, cx_sd_t = cv2.meanStdDev(cx_t, mask_t)
    cy_mean_t, cy_sd_t = cv2.meanStdDev(cy_t, mask_t)
    RBC = D_Bs - D_Rs #吸光度の差分計算
    RBC_mean_s, RBC_sd_s = cv2.meanStdDev(RBC, mask_s)
    #変換先の色分布へ補正
    weight = np.square(np.clip(((D_Bs - D_Rs) - RBC_mean_s) / (2 * RBC_sd_s), 0, 1))
    #weight = np.clip(((D_Bs - D_Rs) - RBC_mean_s - RBC_sd_s) / (2 * RBC_sd_s), 0, 1)
    #D_c = (D_s - D_mean_s[0]) * D_sd_t[0] / D_sd_s[0] + D_mean_t[0]
    weight_t = np.where(mask_s == 255, weight, 1)
    D_c = D_s
    cx_c = (cx_s - cx_mean_s[0]) * cx_sd_t[0] / cx_sd_s[0] + cx_mean_t[0]
    cy_c = (cy_s - cy_mean_s[0]) * cy_sd_t[0] / cy_sd_s[0] + cy_mean_t[0]
    D_c = weight_t * D_s + (1 - weight_t) * D_c
    cx_c = weight_t * cx_s + (1 - weight_t) * cx_c
    cy_c = weight_t * cy_s + (1 - weight_t) * cy_c
    D_Rc = D_c * (cx_c + 1)
    D_Gc = D_c * (2 - cx_c + math.sqrt(3) * cy_c) / 2
    D_Bc = D_c * (2 - cx_c - math.sqrt(3) * cy_c) / 2
    #R_c = np.where(mask_s == 255, np.exp(-D_Rc) * R_max_t, R_max_t)
    #G_c = np.where(mask_s == 255, np.exp(-D_Gc) * G_max_t, G_max_t)
    #B_c = np.where(mask_s == 255, np.exp(-D_Bc) * B_max_t, B_max_t)
    R_c = np.exp(-D_Rc) * R_max_t
    G_c = np.exp(-D_Gc) * G_max_t
    B_c = np.exp(-D_Bc) * B_max_t
    R_cvt = Image.fromarray(np.uint8(np.clip(R_c, 0.0, 255.0)))
    G_cvt = Image.fromarray(np.uint8(np.clip(G_c, 0.0, 255.0)))
    B_cvt = Image.fromarray(np.uint8(np.clip(B_c, 0.0, 255.0)))
    img_cvt = Image.merge('RGB', (R_cvt,G_cvt,B_cvt))
    img_cvt.save('./thumb_cn/'+src)

######## main ########

files = os.listdir('./svs')
for i in range(len(files)):
    #SVSファイルオープン
    img = openslide.OpenSlide('./svs/'+files[i])
    width,height = img.dimensions
    #ディレクトリを確認し，存在しなければ作成
    dn = files[i].rstrip('.svs')    #ディレクトリ名取得
    if not os.path.exists('./patch/'+dn):
        os.mkdir('./patch/'+dn)
        #svsの画素数からブロック画像枚数を計算
    i_num = [] #画像数配列
    i_num.append(width//b_size) #水平方向画像数
    i_num.append(height//b_size) #垂直方向画像数

    #テキストに画像枚数を書き出し
    f = open('./patch/'+dn+'/i_num.txt', 'w')
    f.write('w_num:\t'+str(i_num[0])+'\nh_num:\t'+str(i_num[1]))
    f.close()

    #画像切り出し
    i = 0
    for ig in DivideSVS(img, i_num[0], i_num[1], b_size):
        ig.save('./patch/'+dn+'/'+dn+'_'+str(i)+ext)
        i = i + 1
    MakeThumb(dn, i_num[0], i_num[1], t_size)

files_p = os.listdir('./patch')
ROIdetection('15322')
for i in range(len(files_p)):
    ROIdetection(files_p[i])
    ColorNorm_patch(files_p[i], '15322') #変換元と任意の変換先
