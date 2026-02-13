import torch
import torch.nn.functional as F

from obspy.signal.invsim import cosine_taper

def convert_to_freq_out(output_in_time, fs, freqmin, freqmax, pad=0, taper=None):
    #output_in_time: (nsrc, nrec, nt, 1), the last dimension includes real vz
    #-> 
    #output_in_freq: (nsrc, nf, nrec, 2), the last dimension includes complex waveforms

    if output_in_time.shape[-2] % 2 == 0:
        output_in_time = F.pad(output_in_time.permute(0, 1, 3, 2), (pad, pad), mode='constant', value=0).permute(0, 1, 3, 2)
    else:
        output_in_time = F.pad(output_in_time.permute(0, 1, 3, 2), (pad, pad+1), mode='constant', value=0).permute(0, 1, 3, 2)
    
    nt = output_in_time.shape[-2]
    freqs = torch.arange(nt // 2 + 1) * fs / (nt - 1)
    freq_to_keep = torch.where((freqs>=freqmin)&(freqs<=freqmax))[0].tolist()

    # output
    if taper is None:
        window = torch.hann_window(output_in_time.size(-2)).unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(
            output_in_time.size(0), 
            output_in_time.size(1), 
            1,
            output_in_time.size(-1)
            )
    else:
        window = torch.from_numpy(cosine_taper(output_in_time.size(-2), p=0.1)).unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(
            output_in_time.size(0), 
            output_in_time.size(1), 
            1,
            output_in_time.size(-1)
            )
    
    output_in_freq = torch.fft.rfft(output_in_time * window, 
                                    dim=-2, norm='backward')  # Negative frequencies omitted 
    output_in_freq = output_in_freq[:, :, freq_to_keep, :]  # (nsrc, nrec, NF, 1)
    output_in_freq = torch.view_as_real(output_in_freq)  # view_as_real operates on the last dimension (nsrc, nrec, NF, 1, 2)
    output_in_freq = output_in_freq.permute(0, 2, 1, 3, 4)  # Move the frenquency domain forward (nsrc, NF, nrec, 1, 2)
    output_in_freq = output_in_freq.flatten(-2, -1)  # Make complex u in the channel dimension (nsrc, NF, nrec, 2)
    return output_in_freq

def return_to_time(data_in_freq, ns, nt, fs, freqmin, freqmax, depad=0):
    #nt: signal length after padding
    #data_in_freq: (nv*ns*nf, nr, 2)
    #->
    #data_in_time: (nv, ns, nr, nt, 1)

    mark = True
    if nt % 2 != 0:
        mark = False
        nt += 1

    freqs = torch.arange(nt // 2 + 1) * fs / (nt - 1)
    freq_to_keep = torch.where((freqs>=freqmin)&(freqs<=freqmax))[0].tolist()
    NF = len(freq_to_keep)

    device = data_in_freq.device
    data_in_time = data_in_freq.view(-1, NF, data_in_freq.size(-2), data_in_freq.size(-1))  # (nv*ns, nf, nrec, 2)
    data_in_time = data_in_time.view(data_in_time.size(0), data_in_time.size(1), data_in_time.size(2), 1, 2)  # (nv*ns, nf, nrec, 1, 2)
    data_in_time = data_in_time.permute(0, 2, 3, 1, 4).contiguous()  # (nv*ns, nrec, 1, nf, 2)
    data_in_time = torch.view_as_complex(data_in_time)  # (nv*ns, nrec, 1, nf)
    kept_freq = torch.zeros(data_in_time.size(0), 
                            data_in_time.size(1), 
                            data_in_time.size(2), 
                            len(freqs), dtype=torch.cfloat, device=device)
    kept_freq[..., freq_to_keep] = data_in_time
    data_in_time = torch.fft.irfft(kept_freq, dim=-1, norm='backward')  #(nv*ns, nrec, 1, nt)
    if mark:
        data_in_time = data_in_time[..., depad:nt-depad]
    else:
        data_in_time = data_in_time[..., depad:nt-depad-1]

    data_in_time = data_in_time.permute(0, 1, 3, 2)  # (nv*ns, nr, nt, 1)
    data_in_time = data_in_time.view(-1, ns, data_in_time.size(1), data_in_time.size(2), data_in_time.size(3))  # (nv, ns, nr, nt, 1)
    return data_in_time
