clear; close all;clc;

N = 224;
num_clases = 4;
orden_show  = [3 4 5 8 9 10];

%Direccion de las etiquetas
dir_etiquetas = 'C:\Users\germa\Documents\Pancake_nuevo\etiquetas\';
%Direccionde las máscaras de tierra
dir_tierras = 'D:\germa\DriveUP\TransferLearningFull\DB\MascaraTierra\';
%Direccion imagenes para traslapar
dir_imagenes = 'D:\germa\DriveUP\TransferLearningFull\UINT8\ImgsValidacionHH\';
%Direccion de las mascaras de pancake generadas por Erika
dir_original = 'D:\germa\DriveUP\TransferLearningFull\DB\ImagenesErika\MASCARAS_selec_VIV\';


%Nombre de las imagenes
im = {'8F1A', '52FC', '4424', 'D07D', 'E94E'};
%Nombre de los modelos para las etiquuetas
model = {'Red04', 'Red06', 'Red08', 'Red15', 'Red17', 'Red18'};
fnum = {'uint8', 'dB16', 'dB32'};

Res_dice = zeros(size(model,2), size(im,2), size(fnum,2));

for fn = 1:size(fnum,2)
    cuenta = 1;
    for mod = 1:size(model,2)

        for ix = 1:size(im,2)

            [mod ix]
            %Cargar la imagen para traslapar
            Imagen = load([dir_imagenes im{ix} '_HH_TN_Cal_dB_UINT8_MAT.mat']).I;
            %Leer las etiquetas
            y_pred = load([dir_etiquetas model{mod} '_' fnum{fn} '_y_pred_' im{ix} '.mat']).y_pred;
            %Clasificación final
            [~,y_pred2] = max(y_pred,[],2);
            
            %Imagen recortada a un tamaño adecuado
            I = Imagen(1:(floor(size(Imagen,1)/N))*N, 1:(floor(size(Imagen,2)/N))*N);
            %Obtener las ventanas
            block_0 = mat2cell(I, repmat(N, [1, floor(size(I,1)/N)]), repmat(N, [1, floor(size(I,2)/N)]));
            %Imagen que tendrá las etiquetas de clasificación
            Iseg = zeros(size(I));
            
            l = 1;
            for k = 1:size(block_0,1)
                for m = 1:size(block_0,2)
                    %Creación de las ventanas
                    cuadro = y_pred2(l,1).*ones(N);
                    Iseg(((k-1)*N)+1:k*N, ((m-1)*N)+1:m*N) = cuadro;
                    l = l+1;
                end
            end
            
            %Usar solo los que tiene la etiqueta de pancake
            Iseg_1 = (Iseg==1);
            
            %Quitar la tierra
            Tierra = load([dir_tierras im{ix} '_LAND.mat']);
            Tierra = Tierra.h;
            Tierra = logical(Tierra(1:(floor(size(Imagen,1)/N))*N, 1:(floor(size(Imagen,2)/N))*N));
            
            %MÁSCARA FINAL DE PANCAKE
            Pancake_final = Iseg_1 & ~Tierra;
            
            %Leer las mascaras objetivo
            original = load([dir_original im{ix} '_MASK.mat']).BW;
            original = original(1:(floor(size(original,1)/N))*N, 1:(floor(size(original,2)/N))*N);
            Res_dice(mod,ix,fn) = dice(original,Pancake_final);
            
            %Truquito para usa el overlay
            if sum(Pancake_final(:)) == 0
                Pancake_final(end,end) = 1;
            end
            
            figure((10*(fn-1))+ix)
            subplot(2,5,[1,2,6,7]), imshow(labeloverlay(mat2gray(I),original))
            title(['Imagen original ' im{ix}])
            subplot(2,5, orden_show(cuenta)), imshow(labeloverlay(mat2gray(I),Pancake_final))
            title(['Segmentación con ' model{mod}])
            
        end    
    cuenta = cuenta+1;
    end
    
end
Res_dice


% %Nombre de los modelos para las etiquuetas
% model = {'Red04', 'Red06', 'Red08', 'Red15','Red17', 'Red18'};
% fnum = {'dB32'};
% Res_dice32 = zeros(size(model,2), size(im,2));
% 
% for fn = 1
%     
%     for mod = 1:size(model,2)
% 
%         for ix = 1:size(im,2)
% 
%             [mod ix]
%             %Cargar la imagen para traslapar
%             Imagen = load([dir_imagenes im{ix} '_HH_TN_Cal_dB_UINT8_MAT.mat']).I;
%             %Leer las etiquetas
%             y_pred = load([dir_etiquetas model{mod} '_' fnum{fn} '_y_pred_' im{ix} '.mat']).y_pred;
%             %Clasificación final
%             [~,y_pred2] = max(y_pred,[],2);
%             
%             %Imagen recortada a un tamaño adecuado
%             I = Imagen(1:(floor(size(Imagen,1)/N))*N, 1:(floor(size(Imagen,2)/N))*N);
%             %Obtener las ventanas
%             block_0 = mat2cell(I, repmat(N, [1, floor(size(I,1)/N)]), repmat(N, [1, floor(size(I,2)/N)]));
%             %Imagen que tendrá las etiquetas de clasificación
%             Iseg = zeros(size(I));
%             
%             l = 1;
%             for k = 1:size(block_0,1)
%                 for m = 1:size(block_0,2)
%                     %Creación de las ventanas
%                     cuadro = y_pred2(l,1).*ones(N);
%                     Iseg(((k-1)*N)+1:k*N, ((m-1)*N)+1:m*N) = cuadro;
%                     l = l+1;
%                 end
%             end
%             
%             %Usar solo los que tiene la etiqueta de pancake
%             Iseg_1 = (Iseg==1);
%             
%             %Quitar la tierra
%             Tierra = load([dir_tierras im{ix} '_LAND.mat']);
%             Tierra = Tierra.h;
%             Tierra = logical(Tierra(1:(floor(size(Imagen,1)/N))*N, 1:(floor(size(Imagen,2)/N))*N));
%             
%             %MÁSCARA FINAL DE PANCAKE
%             Pancake_final = Iseg_1 & ~Tierra;
%             
%             %Leer las mascaras objetivo
%             original = load([dir_original im{ix} '_MASK.mat']).BW;
%             original = original(1:(floor(size(original,1)/N))*N, 1:(floor(size(original,2)/N))*N);
%             Res_dice32(mod,ix) = dice(original,Pancake_final);
%             
%         end    
% 
%     end
%     Res_dice32
% end
