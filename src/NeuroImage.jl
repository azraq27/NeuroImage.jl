module NeuroImage

export Dataset
export loadnifti,savenifti,data,affine,size
export ijk_to_xyz,xyz_to_ijk

using PyCall

type Dataset
    fname::String
    pyobj::PyObject
end

Dataset{T<:Real,N,K<:Real}(fname::String,a::AbstractArray{T,N},aff::Array{K,2}) = Dataset(fname,nib.Nifti1Image(a,aff))
Dataset{T<:Real,N}(fname::String,a::AbstractArray{T,N},template::Dataset) = Dataset(fname,nib.Nifti1Image(a,affine(template)))

@pyimport nibabel as nib

function loadnifti(fname::String)
    img = nib.load(fname)
    return Dataset(fname,img)
end

function Dataset{T<:Real}(d::Dataset,a::AbstractArray{T,3})
    img = nib.Nifti1Image(a,d.pyobj[:affine],d.pyobj[:header])
    Dataset(d.fname,img)
end

function savenifti(d::Dataset,fname::String)
    nib.save(d.pyobj,fname)
end

save(d::Dataset) = save(d,d.fname)

function data(d::Dataset)
    d.pyobj[:get_data]()
end

function affine(d::Dataset)
    d.pyobj[:affine]
end

import Base.size
function size(d::Dataset)
    d.pyobj[:header][:get_data_shape]()
end

function ijk_to_xyz{T<:Real}(d::Dataset,ijk::AbstractArray{T,1})
    M = affine(d)[1:3,1:3]
    abc = affine(d)[1:3, 4]
    squeeze(M*reshape([Float64(x) for x in ijk],(3,1)) + abc,2)
end

function xyz_to_ijk{T<:Real}(d::Dataset,xyz::AbstractArray{T,1})
    M = affine(d)[1:3,1:3]
    abc = affine(d)[1:3, 4]
    [Int(round(x)) for x in M \ (xyz .- abc)]
end

using Images

function Base.show(io::IO, mime::MIME"image/png", d::Dataset)
    sd = size(d)
    s = Int(round(sd[1]/3))
    dd = 0
    if length(size(d))==4
        dd = Array{Float32,2}(NeuroImage.data(d)[s,end:-1:1,end:-1:1,1])'
    else
        dd = Array{Float32,2}(NeuroImage.data(d)[s,end:-1:1,end:-1:1,1])'
    end
    dd ./= maximum(dd)
    show(io,mime,colorview(Gray,dd))
end

end
