function varargout = untitled1(varargin)
% UNTITLED1 MATLAB code for untitled1.fig
%      UNTITLED1, by itself, creates a new UNTITLED1 or raises the existing
%      singleton*.
%
%      H = UNTITLED1 returns the handle to a new UNTITLED1 or the handle to
%      the existing singleton*.
%
%      UNTITLED1('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in UNTITLED1.M with the given input arguments.
%
%      UNTITLED1('Property','Value',...) creates a new UNTITLED1 or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before untitled1_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to untitled1_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help untitled1

% Last Modified by GUIDE v2.5 01-Jan-2025 11:17:53

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @untitled1_OpeningFcn, ...
                   'gui_OutputFcn',  @untitled1_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before untitled1 is made visible.
function untitled1_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to untitled1 (see VARARGIN)

% Choose default command line output for untitled1
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes untitled1 wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = untitled1_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
%打开
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to load (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
  [file path]=uigetfile({'*.jpg';'*.bmp';'*.jpeg';'*.png'}, '打开文件');%uigetfile图像用户界面模块
image=[path file];

handles.file=image;
if (file==0)
    warndlg('请选择一张图片...') ;
end
[fpath, fname, fext]=fileparts(file);
validex=({'.bmp','.jpg','.jpeg','.png'});
found=0;
for (x=1:length(validex))
   if (strcmpi(fext,validex{x}))
       found=1;
     

handles.img=imread(image);
handles.i=imread(image);
h = waitbar(0,'等待...');
steps = 100;

for step = 1:steps
    waitbar(step / steps)
end
close(h) 
axes(handles.axes2); 
cla; 
imshow(handles.img);
axes(handles.axes4); 
cla; 
imshow(handles.img);
guidata(hObject,handles);
break; 
end
end
if (found==0)
     errordlg('文件扩展名不正确，请从可用扩展名[.jpg、.jpeg、.bmp、.png]中选择文件','Image Format Error');
end

% --- Executes on button press in exit.
function exit_Callback(hObject, eventdata, handles)
% hObject    handle to exit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
close all;
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

%灰度化
% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
img =handles.i;
img = rgb2gray(img);
size(img)
axes(handles.axes4);
imshow(img)

%直方图均衡化
% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
    
    % 检查图像是否为彩色，如果是彩色图像，转换为灰度图
    mysize = size(handles.img);
    if numel(mysize) > 2
        handles.img = rgb2gray(handles.img);
    end

    % 对图像进行直方图均衡化
    handles.img = histeq(handles.img);

    % 在指定的 axes4 中显示均衡化后的图像
    axes(handles.axes4);
    cla; % 清空当前坐标轴内容
    imshow(handles.img); % 显示均衡化后的图像

    % 更新句柄数据
  guidata(hObject,handles);


%soble
% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to m1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
mysize=size(handles.img);
if numel(mysize)>2
    handles.img=rgb2gray(handles.img);
end
handles.img=edge(handles.img,'sobel');
axes(handles.axes4);
cla;
imshow(handles.img);
guidata(hObject,handles);


%高斯滤波
% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
  hsize=[8 8]; sigma=3;%hsize是滤波器尺寸，sigma是标准差
h=fspecial('gaussian',hsize,sigma);%fspecial生成滤波器
handles.img=imfilter(handles.img,h,'replicate');%imfilter滤波，控制滤波运算的选项类型'replicate'
axes(handles.axes4); cla; imshow(handles.img);
guidata(hObject,handles);

%中值滤波
% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
r=medfilt2(handles.img(:,:,1));%执行二维中值滤波
g=medfilt2(handles.img(:,:,2));
b=medfilt2(handles.img(:,:,3)); 
handles.img=cat(3,r,g,b);%cat构造三维数组
axes(handles.axes4); cla; imshow(handles.img);
guidata(hObject,handles);
%高斯噪声
% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
 prompt = {'请输入高斯噪声的均值（默认 0）:', '请输入高斯噪声的方差（默认 0.01）:'};
    dlgTitle = '高斯噪声参数配置';
    dims = [1 35];
    defInput = {'0', '0.01'}; % 默认均值为 0，方差为 0.01
    answer = inputdlg(prompt, dlgTitle, dims, defInput);

    if isempty(answer)
        return; % 如果用户取消输入，则退出
    end

    % 转换用户输入为数值
    noiseMean = str2double(answer{1});
    noiseVariance = str2double(answer{2});

    % 检查参数合法性
    if isnan(noiseMean) || isnan(noiseVariance) || noiseVariance < 0
        errordlg('请输入有效的噪声均值和方差！', '错误');
        return;
    end

    % 向图像添加高斯噪声
    noisyImage = imnoise(handles.img, 'gaussian', noiseMean, noiseVariance);

    % 显示带高斯噪声的图像
    axes(handles.axes4);
    cla; % 清空坐标轴
    imshow(noisyImage);
    title(sprintf('添加高斯噪声 (均值: %.2f, 方差: %.2f)', noiseMean, noiseVariance));

    % 更新句柄数据
    handles.noisyImg = noisyImage; % 保存带噪声的图像
    guidata(hObject, handles); % 保存更新后的句柄数据

%旋转
% --- Executes on button press in pushbutton9.
function pushbutton9_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
  prompt = {'请输入旋转角度（正数为顺时针，负数为逆时针，单位为度）:'};
    dlgTitle = '旋转图像';
    dims = [1 35];
    defInput = {'0'}; % 默认角度为 0
    answer = inputdlg(prompt, dlgTitle, dims, defInput);

    if isempty(answer)
        return; % 如果用户取消输入，则退出
    end

    % 转换用户输入为数值
    rotationAngle = str2double(answer{1});
    if isnan(rotationAngle)
        errordlg('请输入有效的角度值！', '错误');
        return;
    end

    % 旋转图像
    rotatedImage = imrotate(handles.img, rotationAngle);

    % 显示旋转后的图像
    axes(handles.axes4);
    cla; % 清空当前坐标轴内容
    imshow(rotatedImage); % 显示旋转后的图像
    title(sprintf('旋转角度: %.2f°', rotationAngle)); % 显示旋转角度

    % 更新句柄数据
    handles.rotatedImg = rotatedImage; % 保存旋转后的图像
    guidata(hObject, handles); % 保存更新后的句柄数据

%缩放
% --- Executes on button press in pushbutton10.
function pushbutton10_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



%指数变换
% --- Executes on button press in pushbutton11.
function pushbutton11_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
 mysize = size(handles.img);
    if numel(mysize) > 2
        handles.img = rgb2gray(handles.img); % 转换为灰度图像
    end

    % 对数变换公式：s = c * log(1 + r)
    % 计算缩放常数 c
    c = 255 / log(1 + double(max(handles.img(:)))); % 动态计算 c 值
    logImage = c * log(1 + double(handles.img));    % 应用对数变换
    logImage = uint8(logImage); % 转换为 uint8 格式，适应图像显示

    % 在指定的 axes4 中显示对数变换后的图像
    axes(handles.axes4);
    cla; % 清空当前坐标轴内容
    imshow(logImage); % 显示对数变换后的图像

    % 更新句柄数据
    handles.img = logImage; % 将对数变换后的图像保存到句柄
    guidata(hObject, handles); % 保存更新后的句柄数据


%对数变换
% --- Executes on button press in pushbutton12.
function pushbutton12_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
mysize = size(handles.img);
    if numel(mysize) > 2
        handles.img = rgb2gray(handles.img); % 转换为灰度图像
    end

    % 对数变换公式：s = c * log(1 + r)
    % 计算缩放常数 c
    c = 255 / log(1 + double(max(handles.img(:)))); % 动态计算 c 值
    logImage = c * log(1 + double(handles.img));    % 应用对数变换
    logImage = uint8(logImage); % 转换为 uint8 格式，适应图像显示

    % 在指定的 axes4 中显示对数变换后的图像
    axes(handles.axes4);
    cla; % 清空当前坐标轴内容
    imshow(logImage); % 显示对数变换后的图像

    % 更新句柄数据
    handles.img = logImage; % 将对数变换后的图像保存到句柄
    guidata(hObject, handles); % 保存更新后的句柄数据


%线性变换
% --- Executes on button press in pushbutton13.
function pushbutton13_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton13 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
 mysize = size(handles.img);
    if numel(mysize) > 2
        handles.img = rgb2gray(handles.img); % 转换为灰度图像
    end

    % 执行线性变换 (自动对比度拉伸)
    handles.img = imadjust(handles.img, stretchlim(handles.img), []);

    % 在指定的 axes4 中显示线性变换后的图像
    axes(handles.axes4);
    cla; % 清空当前坐标轴内容
    imshow(handles.img); % 显示线性变换后的图像

    % 更新句柄数据
    guidata(hObject, handles);

   %椒盐噪声
% --- Executes on button press in pushbutton8.
function pushbutton8_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
 mysize = size(handles.img);
    if numel(mysize) > 2
        img = rgb2gray(handles.img); % 转换为灰度图像
    else
        img = handles.img;
    end

    % 默认椒盐噪声密度
    noiseDensity = 0.05; % 默认噪声密度为 5%

    % 用户输入噪声密度
    prompt = {'请输入椒盐噪声密度（0~1，默认：0.05）:'};
    dlgTitle = '添加椒盐噪声';
    dims = [1 35];
    defInput = {'0.05'}; % 默认值
    answer = inputdlg(prompt, dlgTitle, dims, defInput);

    if isempty(answer)
        return; % 如果用户取消输入，则退出
    end

    % 获取用户输入的噪声密度
    noiseDensity = str2double(answer{1});
    if isnan(noiseDensity) || noiseDensity <= 0 || noiseDensity > 1
        errordlg('噪声密度必须是 0 到 1 之间的数值！', '错误');
        return;
    end

    % 添加椒盐噪声
    noisyImage = imnoise(img, 'salt & pepper', noiseDensity);

    % 在指定的 axes4 中显示带噪声的图像
    axes(handles.axes4);
    cla; % 清空当前坐标轴内容
    imshow(noisyImage); % 显示带噪声的图像

    % 更新句柄数据
    handles.noisyImg = noisyImage; % 将带噪声的图像保存到句柄
    guidata(hObject, handles); % 保存更新后的句柄数据


    

%roberts
% --- Executes on button press in pushbutton14.
function pushbutton14_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
mysize=size(handles.img);
if numel(mysize)>2
    handles.img=rgb2gray(handles.img);
end
handles.img=edge(handles.img,'roberts');
axes(handles.axes4);
cla;
imshow(handles.img);
guidata(hObject,handles);

%高通滤波
% --- Executes on button press in pushbutton15.
function pushbutton15_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton15 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
mysize=size(handles.img);
if numel(mysize)>2
    handles.img = rgb2gray(handles.img);
end

I=handles.img;
I=im2double(I);
M=2*size(I,1);%滤波器行数
N=2*size(I,2);%滤波器列数
u=-M/2:(M/2-1);
v=-N/2:(N/2-1);
[U,V]=meshgrid(u,v);
D=sqrt(U.^2+V.^2);
D0=30;%截止频率
n=6;%巴特沃斯滤波器阶数
H=1./(1+(D0./D).^(2*n));
J=fftshift(fft2(I,size(H,1),size(H,2)));
K=J.*H;
L=ifft2(ifftshift(K));
L=L(1:size(I,1),1:size(I,2));
handles.img=L;

axes(handles.axes4); 
cla; 
imshow(handles.img);
guidata(hObject,handles);

%低通滤波
% --- Executes on button press in pushbutton16.
function pushbutton16_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton16 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
mysize=size(handles.img);
if numel(mysize)>2
    handles.img = rgb2gray(handles.img);
end

I=handles.img;
I=im2double(I);
M=2*size(I,1);
N=2*size(I,2);
u=-M/2:(M/2-1);
v=-N/2:(N/2-1);
[U,V]=meshgrid(u,v);
D=sqrt(U.^2+V.^2);
D0=80;
H=double(D<=D0);
J=fftshift(fft2(I,size(H,1),size(H,2)));
K=J.*H;
L=ifft2(ifftshift(K));
L=L(1:size(I,1),1:size(I,2));
handles.img=L;

axes(handles.axes4); 
cla; 
imshow(handles.img);
guidata(hObject,handles);

%均值滤波
% --- Executes on button press in pushbutton17.
function pushbutton17_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton17 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% 检查图像是否为灰度图，如果是彩色图像，转换为灰度图
    mysize = size(handles.img);
    if numel(mysize) > 2
        grayImage = rgb2gray(handles.img); % 转换为灰度图像
    else
        grayImage = handles.img;
    end

    % 添加椒盐噪声
    noiseDensity = 0.05; % 设置椒盐噪声密度
    noisyImage = imnoise(grayImage, 'salt & pepper', noiseDensity);

    % 保存带噪声的图像到句柄
    handles.noisyImg = noisyImage;

  

    % 创建均值滤波器
    filterSize = 3; % 设置均值滤波器尺寸
    h = fspecial('average', filterSize); % 创建均值滤波器

    % 对带噪声的图像应用均值滤波
    filteredImage = imfilter(noisyImage, h, 'replicate'); % 使用滤波器处理图像

    % 显示滤波后的图像
    axes(handles.axes4); % 显示在 axes4 上
    cla;
    imshow(filteredImage);
    title(sprintf('均值滤波 (滤波器尺寸: %dx%d)', filterSize, filterSize));

    % 更新句柄数据
    handles.filteredImg = filteredImage; % 保存滤波后的图像到句柄
    guidata(hObject, handles); % 保存更新后的句柄数据

%prewitt
% --- Executes on button press in pushbutton18.
function pushbutton18_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton18 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
 % 检查图像是否为灰度图，如果是彩色图像，转换为灰度图
    img = handles.img;
    if size(img, 3) > 1
        img = rgb2gray(img); % 转换为灰度图像
    end

    % 定义 Prewitt 水平和垂直算子
    prewittX = [-1 0 1; -1 0 1; -1 0 1]; % 水平方向
    prewittY = [-1 -1 -1; 0 0 0; 1 1 1]; % 垂直方向

    % 应用 Prewitt 算子
    edgeX = imfilter(double(img), prewittX, 'replicate'); % 水平方向边缘
    edgeY = imfilter(double(img), prewittY, 'replicate'); % 垂直方向边缘

    % 组合两个方向的边缘
    edgeImage = sqrt(edgeX.^2 + edgeY.^2);

    % 将边缘检测结果归一化以便显示
    edgeImage = mat2gray(edgeImage);

    % 显示原始图像和边缘检测结果

    axes(handles.axes4);
    cla;
    imshow(edgeImage);
    title('Prewitt 边缘检测');

    % 更新句柄数据
    handles.edgeImg = edgeImage; % 保存边缘检测结果
    guidata(hObject, handles); % 保存更新后的句柄数据

%拉普拉斯
% --- Executes on button press in pushbutton19.
function pushbutton19_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton19 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
 % 检查图像是否为灰度图，如果是彩色图像，转换为灰度图
    img = handles.img;
    if size(img, 3) > 1
        img = rgb2gray(img); % 转换为灰度图像
    end

    % 定义拉普拉斯算子
    laplacianKernel = [0 -1 0; -1 4 -1; 0 -1 0]; % 拉普拉斯模板

    % 应用拉普拉斯算子
    edgeImage = imfilter(double(img), laplacianKernel, 'replicate'); % 计算边缘

    % 将边缘检测结果归一化以便显示
    edgeImage = mat2gray(edgeImage);

    % 显示原始图像和边缘检测结果
   

    axes(handles.axes4);
    cla;
    imshow(edgeImage);
    title('拉普拉斯边缘检测');

    % 更新句柄数据
    handles.edgeImg = edgeImage; % 保存边缘检测结果
    guidata(hObject, handles); % 保存更新后的句柄数据


%HOG
% --- Executes on button press in pushbutton20.
function pushbutton20_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton20 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
I=rgb2gray(handles.img);   %读取到一张图片  
%设置初始阈值   去最大值和最小值的中间值
zmax=max(max(I));
zmin=min(min(I));
T1=(zmax+zmin)/2;
%根据阈值将图像进行分割为前景和背景，分别求出两者的平均灰度  z1和z2
b=1;
[m n]=size(I);
while (b)
        ifg=0;
        ibg=0;
        fnum=0;
        bnum=0;
        for i=1:m
            for j=1:n
                tmp=I(i,j);
                if(tmp>=T1)
                    ifg=ifg+1;
                    fnum=fnum+double(tmp);  %前景像素的个数以及像素值的总和
                else
                    ibg=ibg+1;
                    bnum=bnum+double(tmp);%背景像素的个数以及像素值的总和
                end
            end
        end
        %计算前景和背景的平均值
        z1=fnum/ifg;
        z2=bnum/ibg;
        if(T1==(uint8((z1+z2)/2)))
            b=0;
        else
            T1=uint8((z1+z2)/2);
        end
        %当阈值不变换时，退出迭代
end
thresh = double(T1)/255;
I1=imbinarize(I,thresh);

axes(handles.axes4)
imshow(I1)    %显示二值化之后的图片

%LBP
% --- Executes on button press in pushbutton21.
function pushbutton21_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton21 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% 检查图像是否为灰度图，如果是彩色图像，转换为灰度图
    img = handles.img;
    if size(img, 3) > 1
        img = rgb2gray(img); % 转换为灰度图像
    end

    % 提取 LBP 特征
    lbpFeatures = extractLBPFeatures(img, 'CellSize', [16 16]);

    % 显示原始图像
  

    % 显示 LBP 特征的直方图
    axes(handles.axes4); % 假设有另一个坐标轴 g3 用于显示特征直方图
    cla;
    bar(lbpFeatures);
    title('LBP 特征直方图');

    % 保存特征向量到句柄
    handles.lbpFeatures = lbpFeatures;
    guidata(hObject, handles); % 保存更新后的句柄数据

%阈值分割
% --- Executes on button press in pushbutton22.
function pushbutton22_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton22 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on slider movement.
function slider1_Callback(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider


% --- Executes during object creation, after setting all properties.
function slider1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

%对数变换
% --- Executes during object creation, after setting all properties.
function pushbutton12_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pushbutton12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
function logTransform_Callback(hObject, eventdata, handles)
   


% --- Executes on mouse press over axes background.
function axes2_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to axes2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes during object creation, after setting all properties.
function axes2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes2

%清除
% --- Executes on button press in pushbutton23.
function pushbutton23_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton23 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.img=handles.i;
axes(handles.axes4); 
cla; 
imshow(handles.img);
guidata(hObject,handles);


% --- Executes on slider movement.
function slider2_Callback(hObject, eventdata, handles)
% hObject    handle to slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider


% --- Executes during object creation, after setting all properties.
function slider2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function slider3_Callback(hObject, eventdata, handles)
% hObject    handle to slider3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
rrv=(get(hObject,'Value'))
handles.sf=handles.img;
handles.sf=imresize(handles.sf,rrv);%
axes(handles.axes4); cla; imshow(handles.sf);
guidata(hObject,handles)
% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider


% --- Executes during object creation, after setting all properties.
function slider3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
