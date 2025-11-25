import os

import numpy as np
import plotly.colors
import plotly.graph_objects as go


class TDSpectrum:
    '''
        Time-Dependent Spectrum

        Parameters
        ----------
        filename : str
            path to spectrum file to read

        Attributes
        ----------
        plot_styles : tuple
            list of supported plot styles
        filename : str
            basename of spectrum file readed (def: 'spectrum.glb')
        type : str
            type of spectrum file readed (def: 'DATA')
        comment : str
            comment inclued in the header of spectrum file readed
        times : ndarray(n)
            array of times
        lambdas : ndarray(m)
            array of wavelengths
        absorb : ndarray(m,n)
            matrix of absorbances

        Properties
        ----------
        n_spec : int
            number of spectra
        n_times : int
            number of times
        n_lambdas : int
            number of wavelengths
        filename_trim : str
            filename with '_t' appended to the basename
        lim_times : tuple
            limits of times (min, max)
        lim_lambdas : tuple
            limits of wavelengths (min, max)
        lim_absorb : tuple
            limits of absorbances (min, max)
    '''

    plot_styles = ('2d-times', '2d-lambdas')

    def __init__(self, filename=None):
        self.times = np.array([])
        self.lambdas = np.array([])
        self.absorb = np.array([])
        self.filename = "spectrum.glb"
        self.type = "DATA"
        self.comment = ""
        if filename is not None:
            self.read(filename)

    def __str__(self) -> str:
        return f"File: {self.filename}\n" + \
               f"Times: {self.n_times} :: {self.lim_times[0]} - {self.lim_times[1]}\n" + \
               f"Lambdas: {self.n_lambdas} :: {self.lim_lambdas[0]} - {self.lim_lambdas[1]}\n" + \
               f"Absorbances: {self.lim_absorb[0]:.6f} - {self.lim_absorb[1]:.6f}\n"

    @property
    def n_spec(self) -> int:
        return self.n_times

    @property
    def n_times(self) -> int:
        return self.times.shape[0]

    @property
    def n_lambdas(self) -> int:
        return self.lambdas.shape[0]

    @property
    def filename_trim(self) -> str:
        return "{}_t{}".format(*os.path.splitext(self.filename))

    @property
    def lim_times(self) -> tuple:
        return (self.times.min(), self.times.max())

    @property
    def lim_lambdas(self) -> tuple:
        return (self.lambdas.min(), self.lambdas.max())

    @property
    def lim_absorb(self) -> tuple:
        return (self.absorb.min(), self.absorb.max())

    @staticmethod
    def _find_ndx(l, v):
        '''Find index of first occurence of starting lowercase value v in list l'''
        v = v.lower()
        return [i for i, x in enumerate(l) if x.lower().startswith(v)][0]

    @staticmethod
    def _gaussian_filter1d(input_array, sigma, truncate=4.0):
        radius = int(truncate * sigma + 0.5)
        x = np.arange(-radius, radius + 1)
        phi_x = np.exp(-0.5 / sigma**2 * x**2)
        phi_x = phi_x / phi_x.sum()
        padded = np.pad(input_array, radius, mode='reflect')
        return np.convolve(padded, phi_x, mode='valid')

    @staticmethod
    def _median_filter1d(input_array, size):
        if size < 1:
            return input_array
        pad_left = size // 2
        pad_right = size - 1 - pad_left
        padded = np.pad(input_array, (pad_left, pad_right), mode='reflect')
        windows = np.lib.stride_tricks.sliding_window_view(padded, size)
        return np.median(windows, axis=1)

    def read(self, filename, filestr="", format="") -> None:
        '''
            Wrapper to read files based on extension/format

            Supported formats: GLB, standard CSV, ProDataCSV, BK3A

            Parameters
            ----------
            filename : str
                path to file to read
            filestr : str, optional
                file itself as a contigous string
                if given, filename is not readed
            format : {'glb', 'csv', 'bk3a}, optional
                format of file to read
                if not specified, the extension of the filename is used
                if the extension is not recognized, assumed to be 'glb'
        '''
        self.__init__()
        # guess format from extension if not specified
        readers = {
            'glb': self._read_glb,
            'csv': self._read_csv,
            'bk3a': self._read_bk3a
        }
        # check filename/filestr and read file
        if not filename and not filestr:
            raise ValueError('Either filename or filestr must be provided')
        self.filename = filename
        if not filestr:
            if not os.path.exists(filename):
                raise FileNotFoundError(f'File not found: {filename}')
            else:
                with open(filename, 'r') as f:
                    filestr = f.read()
        # assign format based on input argument/file extension/default
        extension = os.path.splitext(filename)[1][1:].lower()
        if format:
            if format.lower() in readers:
                format = format.lower()
            else:
                raise ValueError(f'Unknown format to read: {format}')
        elif extension in readers:
            format = extension
        else:
            format = list(readers.keys())[0]
        # read file based on format
        readers[format](filestr)

    def _read_glb(self, filestr) -> None:
        '''Read a spectrum from a GLB file'''
        data = [line.strip() for line in filestr.splitlines() if line.strip()]
        # parse header (delimited by '/')
        slash_ndx = self._find_ndx(data, '/')
        header = data[:slash_ndx]
        data = data[slash_ndx + 1:]
        self.type = header[self._find_ndx(header,
                                          'type>')].split('>')[1].strip()
        comment_init_ndx = self._find_ndx(header, '%')
        comment_end_ndx = self._find_ndx(reversed(header), '%')
        self.comment = "\n".join(header[comment_init_ndx + 1:len(header) -
                                        comment_end_ndx - 1])
        # get dimensions
        n_spec = int(data[self._find_ndx(data, 'n_spec')].split()[1])
        n_lamb = int(data[self._find_ndx(data, 'n_lam')].split()[1])
        # times
        time_ndx = int(self._find_ndx(data, 'times:')) + 1
        self.times = np.array(data[time_ndx:time_ndx + n_spec], dtype=float)
        # wavelengths
        lamb_ndx = int(self._find_ndx(data, 'lambda:')) + 1
        self.lambdas = np.array(data[lamb_ndx:lamb_ndx + n_lamb], dtype=float)
        # absorbances
        self.absorb = np.zeros((self.n_times, self.n_lambdas), dtype=float)
        abs_ndx = int(self._find_ndx(data, 'data:')) + 1
        data = data[abs_ndx:]
        for i in range(self.n_times):
            self.absorb[i, :] = np.array(data[i * self.n_lambdas:(i + 1) *
                                              self.n_lambdas],
                                         dtype=float)

    def _read_csv(self, filestr) -> None:
        '''Read a spectrum from a CSV file'''
        data = filestr.splitlines()
        if data[0].lower().startswith('prodatacsv'):
            # ProDataCSV
            data = data[self._find_ndx(data, 'data:') + 1:]
            data = data[self._find_ndx(data, ','):]
            self.times = np.array(
                [i.strip() for i in data.pop(0).split(',') if i.strip()],
                dtype=float)
            self.absorb = np.empty((0, self.n_times), dtype=float)
            for row in data:
                if not row.strip():
                    break
                row = row.split(',')
                self.lambdas = np.append(self.lambdas, float(row[0]))
                self.absorb = np.vstack(
                    (self.absorb, np.array(row[1:], dtype=float)))
            self.absorb = self.absorb.T
        else:
            # standard CSV
            if not data[0].lower().startswith(','):
                del data[0]
            self.lambdas = np.array(
                [i.strip() for i in data.pop(0).split(',') if i.strip()],
                dtype=float)
            self.absorb = np.empty((0, self.n_lambdas), dtype=float)
            for row in data:
                row = row.split(',')
                self.times = np.append(self.times, float(row[0]))
                self.absorb = np.vstack(
                    (self.absorb, np.array(row[1:], dtype=float)))

    def _read_bk3a(self, filestr) -> None:
        '''Read a spectrum from a BK3A file (Bio-Kine 3D Text File from BioLogic)'''
        data = filestr.splitlines()
        # parse header (until "_DATA")
        data_ndx = self._find_ndx(data, '\"_DATA\"')
        header = data[:data_ndx + 1]
        data = data[data_ndx + 1:]
        self.comment = "\n".join(header)
        # get bk3a sub-format
        format_ndx = self._find_ndx(header, '\"_FORMAT\"')
        bk3a_format = header[format_ndx].split()[1].strip('\"')
        # replace commas with dots
        data = [line.replace(',', '.') for line in data]
        # read data based on format
        match bk3a_format:
            case 'MATRIX':
                self.lambdas = np.array(data.pop(0).split()[1:], dtype=float)
                data = np.array([i.split() for i in data], dtype=float)
                self.times = data[:, 0]
                self.absorb = data[:, 1:]
            case 'WTV':
                data = np.array([i.split() for i in data], dtype=float)
                self.lambdas = np.unique(data[:, 0])
                self.times = np.unique(data[:, 1])
                self.absorb = data[:, 2].reshape(
                    (self.n_lambdas, self.n_times)).T
            case 'TWV':
                data = np.array([i.split() for i in data], dtype=float)
                self.times = np.unique(data[:, 0])
                self.lambdas = np.unique(data[:, 1])
                self.absorb = data[:, 2].reshape(
                    (self.n_times, self.n_lambdas))
            case _:
                raise ValueError(f'Unknown bk3a sub-format: {bk3a_format}')

    def formatted_string(self, format="glb") -> str:
        '''
            Wrapper to format the spectrum as a string based on extension/format

            Supported formats: GLB, standard CSV, BK3A

            Parameters
            ----------
            format : {'glb', 'csv', 'bk3a'}, optional
                formatting of the string
                if not specified, assumed to be 'glb'

            Returns
            -------
            str
                content of the file as a string
        '''
        # guess format from extension if not specified
        formatters = {
            'glb': self._string_glb,
            'csv': self._string_csv,
            'bk3a': self._string_bk3a
        }
        if format:
            if format.lower() in formatters:
                format = format.lower()
            else:
                raise ValueError(f'Unknown format to write: {format}')
        else:
            format = list(formatters.keys())[0]
        return formatters[format]()

    def _string_glb(self) -> None:
        '''Get spectrum as GLB string'''
        # header
        data = "APL-ASCII-SPECTRAKINETIC\n"
        data += f"TYPE>{self.type}\n%\n{self.comment}\n%\n/\n"
        data += f"N_spec: {self.n_spec:d}\nN_lam: {self.n_lambdas:d}"
        # times
        data += "\nTimes:\n"
        data += '\n'.join([f'{i:.6f}' for i in self.times])
        # wavelengths
        data += "\nLambda:\n"
        data += '\n'.join([f'{i:.3f}' for i in self.lambdas])
        # absorbances
        data += "\nData:\n"
        for i in range(self.n_times):
            data += '\n'.join([f'{j:.6f}' for j in self.absorb[i, :]])
            data += '\n\n'
        return data

    def _string_csv(self) -> None:
        '''Get spectrum as standard CSV string'''
        # header
        data = "SPECTRA\n"
        # wavelengths header
        data += ',' + ','.join([f'{i:.3f}' for i in self.lambdas]) + '\n'
        # times and absorbances matrix
        for i in range(self.n_times):
            data += f"{self.times[i]:.6f}," + ','.join(
                [f'{j:.6f}' for j in self.absorb[i, :]]) + '\n'
        return data

    def _string_bk3a(self) -> None:
        '''Get spectrum as BK3A string (Bio-Kine 3D Text File from BioLogic) (in MATRIX sub-format)'''
        # header
        data = self.comment + '\n'
        # sub-format specification
        data.replace('WTV', 'MATRIX')
        data.replace('TWV', 'MATRIX')
        # wavelengths header
        data += 'nm\t' + '\t'.join([f'{i:.3f}' for i in self.lambdas]) + '\n'
        # times and absorbances matrix
        for i in range(self.n_times):
            data += f"{self.times[i]}\t" + '\t'.join(
                [f'{j:.6f}' for j in self.absorb[i, :]]) + '\n'
        return data

    def plot(self, style='2d-times') -> None:
        '''
            Display a plot of the spectra

            Parameters
            ----------
            style : str, optional
                type of plot to draw (def: '2d-times')
                '2d-times' : multiple superposed spectra (times)
                             wavelength (x) vs. absorbance (y)
                '2d-lambdas' : multiple superposed spectra (wavelengths)
                               time (x) vs. absorbance (y)
                '3d' : 3D plot of spectra [not implemented]
                       time (x) vs. wavelength (y) vs. absorbance (z)
        '''
        n_times = self.n_times
        n_lambdas = self.n_lambdas

        color_palette = plotly.colors.sequential.Viridis

        fig = go.Figure()

        if n_times == 0 or n_lambdas == 0:
            fig.update_layout(title="No data to plot")
            return fig

        match style:
            case '2d-times':
                fig.update_layout(title=self.filename,
                                  xaxis_title='Wavelength (λ)',
                                  yaxis_title='Absorbance',
                                  width=800,
                                  height=400,
                                  xaxis=dict(range=self.lim_lambdas),
                                  yaxis=dict(range=self.lim_absorb))

                for i in range(n_times):
                    color_idx = int((i / n_times) * (len(color_palette) - 1))
                    color = color_palette[color_idx]
                    fig.add_trace(
                        go.Scatter(x=self.lambdas,
                                   y=self.absorb[i, :],
                                   mode='lines',
                                   line=dict(color=color),
                                   name=f'{self.times[i]:.2f} s'))
            case '2d-lambdas':
                fig.update_layout(title=self.filename,
                                  xaxis_title='Time',
                                  yaxis_title='Absorbance',
                                  width=800,
                                  height=400,
                                  xaxis=dict(range=self.lim_times),
                                  yaxis=dict(range=self.lim_absorb))
                for i in range(n_lambdas):
                    color_idx = int((i / n_lambdas) * (len(color_palette) - 1))
                    color = color_palette[color_idx]
                    fig.add_trace(
                        go.Scatter(x=self.times,
                                   y=self.absorb[:, i],
                                   mode='lines',
                                   line=dict(color=color),
                                   name=f'{self.lambdas[i]:.2f} nm'))
            case '3d':
                fig.update_layout(title=self.filename,
                                  scene=dict(
                                      xaxis_title='Time',
                                      yaxis_title='Wavelength (λ)',
                                      zaxis_title='Absorbance',
                                      xaxis=dict(range=self.lim_times),
                                      yaxis=dict(range=self.lim_lambdas),
                                      zaxis=dict(range=self.lim_absorb)),
                                  width=800,
                                  height=600)
                for i in range(n_times):
                    color_idx = int((i / n_times) * (len(color_palette) - 1))
                    color = color_palette[color_idx]
                    fig.add_trace(
                        go.Scatter3d(x=[self.times[i]] * n_lambdas,
                                     y=self.lambdas,
                                     z=self.absorb[i, :],
                                     mode='lines',
                                     line=dict(color=color),
                                     name=f'{self.times[i]:.2f} s'))
            case _:
                raise ValueError(f'Unknown style to plot: {style}')
        return fig

    def zero(self, lamb) -> None:
        '''Modify absorbances to make zero at a specific wavelength'''
        lamb_ndx = np.argmin(abs(self.lambdas - lamb))
        for i in range(self.n_times):
            self.absorb[i, :] -= self.absorb[i, lamb_ndx]

    def trim(self, time=[], lamb=[]) -> None:
        '''
            Trim spectra to a specific time/wavelength range

            Parameters
            ----------
            time : list, optional
                min and max time to keep
            lamb : list, optional
                min and max wavelength to keep
        '''
        # trim times
        if time:
            if len(time) != 2:
                raise ValueError("'time' must be a list of length 2")
            time.sort()
            time_ndx_min = np.argmin(abs(self.times - time[0]))
            time_ndx_max = np.argmin(abs(self.times - time[1]))
            self.times = self.times[time_ndx_min:time_ndx_max + 1]
            self.absorb = self.absorb[time_ndx_min:time_ndx_max + 1]
        # trim wavelengths
        if lamb:
            if len(lamb) != 2:
                raise ValueError("'lamb' must be a list of length 2")
            lamb.sort()
            lamb_ndx_min = np.argmin(abs(self.lambdas - lamb[0]))
            lamb_ndx_max = np.argmin(abs(self.lambdas - lamb[1]))
            self.lambdas = self.lambdas[lamb_ndx_min:lamb_ndx_max + 1]
            self.absorb = self.absorb[:, lamb_ndx_min:lamb_ndx_max + 1]

    def smooth(self, method='sma', scale=1, **kwargs) -> None:
        '''
            Smooth spectra wavelenghts for all times

            Parameters
            ----------
            method : str, optional
                type of smoothing to apply (def: 'gaussian')
                'gaussian' : Gaussian filter
                    'sigma' : standard deviation (def: 1)
                'sma' : simple moving average, trims edges
                    'window' : window size to take average (def: 5)
                'median' : median filter
                    'size' : size to apply filter (def: 2)
            scale : float, optional
                multiplier to scalate the smoothing parameter (def: 1)
        '''
        # default kwargs options
        kwargs_def = {'sigma': 1, 'window': 5, 'size': 2}
        kwargs = {**kwargs_def, **kwargs}
        # apply smoothing
        match method.lower():
            case 'gaussian':
                for i in range(self.n_times):
                    self.absorb[i, :] = self._gaussian_filter1d(
                        self.absorb[i, :], kwargs['sigma'] * scale)
            case 'sma':
                window = int(kwargs['window'] * scale)
                if window >= self.n_lambdas // 2:
                    raise ValueError(
                        "'window' must be smaller than half the number of wavelengths"
                    )
                for i in range(self.n_times):
                    self.absorb[i, :] = np.convolve(self.absorb[i, :],
                                                    np.ones(window) / window,
                                                    mode='same')
                self.lambdas = self.lambdas[window:-window]
                self.absorb = self.absorb[:, window:-window]
            case 'median':
                size = int(kwargs['size'] * scale)
                size = size if size > 1 else 1
                for i in range(self.n_times):
                    self.absorb[i, :] = self._median_filter1d(
                        self.absorb[i, :], size)
            case _:
                raise ValueError(f'Unknown smoothing method: {method}')
