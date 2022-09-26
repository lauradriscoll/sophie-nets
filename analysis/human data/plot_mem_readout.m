function plot_mem_readout(B1,B2,allCR,tinds,col)

D = cat(2,B1,B2);
[Q,~] = qr(D); 
CosTheta = dot(B1,B2)/(norm(B1)*norm(B2));
ThetaInDegrees = acosd(CosTheta)

cmap = colormap(lines(6));
for t = tinds
    use = squeeze(allCR(t,:,:));
    p4 = plot(use*Q(:,1),use*Q(:,2),'color', cmap(col(t)+2,:),'linewidth',2);
    p4.Color(4) = 0.25;
    p4 = plot(use(1,:)*Q(:,1),use(1,:)*Q(:,2),'^','color', cmap(col(t)+2,:),'linewidth',2);
%     p4 = plot(use(end,:)*Q(:,1),use(end,:)*Q(:,2),'o','color', cmap(col(t)+2,:),'linewidth',2);
end