module NeuroImage

export Dataset
export loadnifti,save,data,affine,size
export cluster
export ijk_to_xyz,xyz_to_ijk

using PyCall

mutable struct Dataset
    fname::String
    pyobj::PyObject
    info::PyObject
    subbricks::Vector{String}
end

Dataset(fname::String,pyobj::PyObject) = Dataset(fname,pyobj,PyObject(nothing),String[])

Dataset(fname::String,a::AbstractArray{T,N},aff::Array{K,2}) where {T<:Real,N,K<:Real} = Dataset(fname,nib.Nifti1Image(a,aff))
Dataset(fname::String,a::AbstractArray{T,N},template::Dataset) where {T<:Real,N} = Dataset(fname,nib.Nifti1Image(a,affine(template)))

@pyimport nibabel as nib
@pyimport neural as nl

function loadnifti(fname::String)
    img = nib.load(fname)
    info = nl.dset_info(fname)
    subbricks = [x["label"] for x in info[:subbricks]]
    return Dataset(fname,img,info,subbricks)
end

function Dataset{T<:Real}(d::Dataset,a::AbstractArray{T,3})
    img = nib.Nifti1Image(a,d.pyobj[:affine],d.pyobj[:header])
    Dataset(d.fname,img)
end

function Dataset{T<:Real}(a::AbstractArray{T,3},fname::String="noname.nii.gz")
    img = nib.Nifti1Image(a,nothing)
    Dataset(fname,img)
end

function save(d::Dataset,fname::String)
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

# AFNI dependent functions

function cluster{T<:Real,K<:Real,L<:Real}(clust_map::Array{T,3},r::K,clust_size::L)
    tmpname = "tmp.nii.gz"
    clustname = "tmp_clust.nii.gz"
    for n in [tmpname,clustname]
        isfile(n) && rm(n)
    end
    temp_dset = Dataset(clust_map,tmpname)
    NeuroImage.save(temp_dset)
    run(pipeline(`3dmerge -1clust $r $clust_size -prefix $clustname $tmpname`,stdout=DevNull,stderr=DevNull))
    newd = data(loadnifti(clustname))
    for n in [tmpname,clustname]
        isfile(n) && rm(n)
    end
    return newd
end

function cluster{K<:Real,L<:Real}(clust_map::BitArray{3},r::K,clust_size::L)
    c = cluster(Array{Int16,3}(clust_map),r,clust_size)
    return c.!=0
end

end
