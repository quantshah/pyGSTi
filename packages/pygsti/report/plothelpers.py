from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Helper Functions for generating plots """

import numpy             as _np
import warnings          as _warnings

from .. import tools     as _tools
from .. import objects   as _objs

from ..objects import smart_cached


def get_gatestring_map(gateString, dataset, strs, fidpair_filter=None,
                       gatestring_filter=None, gateLabelAliases=None):
    """ 
    Pre-compute a list of (i,j,gstr) tuples for use in other matrix-
    generation functions.  

    This consolidates all the logic for selecting a subset (via fidpairs_filter,
    gatestring_filter, or  dataset membership) of prep + base + effect
    strings to compute.  The element (i,j,gstr) means that the (i,j)-th
    element of a resulting matrix corresponds to the gate string gstr.
    Typically gstr = prep[j] + gateString + effect[i].  Matrix indices that
    are absent correspond to Nan entries in a resulting matrix.

    Parameters
    ----------
    gateString : tuple of gate labels
        The base gate sequence that is sandwiched between each effectStr
        and prepStr.

    dataset : DataSet
        The data used to test for gate sequence membership

    strs : 2-tuple
        A (prepStrs,effectStrs) tuple usually generated by calling get_spam_strs(...)

    fidpair_filter : list, optional
        If not None, a list of (iRhoStr,iEStr) tuples specifying a subset of
        all the prepStr,effectStr pairs to include in a result matrix.

    gatestring_filter : list, optional
        If not None, a list of GateString objects specifying which elements of
        result matrices should be computed.  Any matrix entry corresponding to
        an gate string *not* in this list is set to NaN.  When both
        fidpair_filter and gatesetring_filter are non-None, gatestring_filter
        is given precedence.

    gateLabelAliases : dictionary, optional
        Dictionary whose keys are gate label "aliases" and whose values are tuples
        corresponding to what that gate label should be expanded into when checking
        for membership in the dataset.


    Returns
    -------
    tuples : list
        A list of (i,j,gstr) tuples.
    rows : int
        The number of rows in resulting matrices
    cols : int
        The number of columns in resulting matrices
    """
    tuples = []
    prepStrs, effectStrs = strs # LEXICOGRAPHICAL VS MATRIX ORDER
    if gateString is None: 
        return tuples, len(effectStrs),len(prepStrs) #all-NaN mxs

    if gatestring_filter is not None:
        gs_filter_dict = { gs: True for gs in gatestring_filter } #fast lookups

    #No filtering -- just fiducial pair check
    for i,effectStr in enumerate(effectStrs):
        for j,prepStr in enumerate(prepStrs):
            gstr = prepStr + gateString + effectStr
            ds_gstr = _tools.find_replace_tuple(gstr,gateLabelAliases)
            if dataset is None or ds_gstr in dataset:
                #Note: gatestring_filter trumps fidpair_filter
                if gatestring_filter is None:
                    if fidpair_filter is None or (j,i) in fidpair_filter:
                        tuples.append((i,j,gstr))
                elif gstr in gs_filter_dict:
                    tuples.append((i,j,gstr))

    return tuples,len(effectStrs),len(prepStrs)



def expand_aliases_in_map(gatestring_map, gateLabelAliases):
    """
    Returns a new gate string map whose strings have been 
    modified to expand any aliases given by `gateLabelAliases`.

    Parameters
    ----------
    gatestring_map : list of tuples
        the original gate string map, typically obtained by 
        calling :func:`get_gatestring_map`.

    gateLabelAliases : dictionary
        Dictionary whose keys are gate label "aliases" and whose values are tuples
        corresponding to what that gate label should be expanded into.
        e.g. gateLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    Returns
    -------
    list
        A list of (i,j,gstr) tuples.
    rows : int
        The number of rows in resulting matrices
    cols : int
        The number of columns in resulting matrices
    """    
    if gateLabelAliases is None: return gatestring_map

    gatestring_tuples, rows, cols = gatestring_map
    
    #find & replace aliased gate labels with their expanded form
    new_gatestring_tuples = []
    for (i,j,s) in gatestring_tuples:
        new_gatestring_tuples.append(
            (i,j, _tools.find_replace_tuple(s,gateLabelAliases)) )

    return new_gatestring_tuples, rows, cols



def total_count_matrix(gsplaq, dataset):
    """
    Computes the total count matrix for a base gatestring.

    Parameters
    ----------
    gsplaq : GatestringPlaquette
        Obtained via :method:`GatestringStructure.get_plaquette`, this object
        specifies which matrix indices should be computed and which gate strings
        they correspond to.

    dataset : DataSet
        The data used to specify the counts

    Returns
    -------
    numpy array of shape (M,N)
        total count values (sum of count values for each SPAM label)
        corresponding to gate sequences where gateString is sandwiched
        between the specified set of N prep-fiducial and M effect-fiducial
        gate strings.
    """
    ret = _np.nan * _np.ones( (gsplaq.rows,gsplaq.cols), 'd')
    for i,j,gstr in gsplaq:
        ret[i,j] = dataset[ gstr ].total()
    return ret



def count_matrices(gsplaq, dataset, spamlabels):
    """
    Computes spamLabel's count matrix for a base gatestring.

    Parameters
    ----------
    gsplaq : GatestringPlaquette
        Obtained via :method:`GatestringStructure.get_plaquette`, this object
        specifies which matrix indices should be computed and which gate strings
        they correspond to.

    dataset : DataSet
        The data used to specify the counts

    spamlabels : list of strings
        The spam labels to extract counts for, e.g. ['plus']

    Returns
    -------
    numpy array of shape ( len(spamlabels), len(effectStrs), len(prepStrs) )
        count values corresponding to spamLabel and gate sequences
        where gateString is sandwiched between the each prep-fiducial and
        effect-fiducial pair.
    """
    ret = _np.nan * _np.ones( (len(spamlabels),gsplaq.rows,gsplaq.cols), 'd')
    for i,j,gstr in gsplaq:
        datarow = dataset[ gstr ]
        ret[:,i,j] = [datarow[sl] for sl in spamlabels]
    return ret


def frequency_matrices(gsplaq, dataset, spamlabels):
    """
    Computes spamLabel's frequency matrix for a base gatestring.

    Parameters
    ----------
    gsplaq : GatestringPlaquette
        Obtained via :method:`GatestringStructure.get_plaquette`, this object
        specifies which matrix indices should be computed and which gate strings
        they correspond to.

    dataset : DataSet
        The data used to specify the frequencies

    spamlabels : list of strings
        The spam labels to extract frequencies for, e.g. ['plus']


    Returns
    -------
    numpy array of shape ( len(spamlabels), len(effectStrs), len(prepStrs) )
        frequency values corresponding to spamLabel and gate sequences
        where gateString is sandwiched between the each prep-fiducial,
        effect-fiducial pair.
    """
    return count_matrices(gsplaq, dataset, spamlabels) \
           / total_count_matrix( gsplaq, dataset)[None,:,:]



def probability_matrices(gsplaq, gateset, spamlabels,
                         probs_precomp_dict=None):
    """
    Computes spamLabel's probability matrix for a base gatestring.

    Parameters
    ----------
    gsplaq : GatestringPlaquette
        Obtained via :method:`GatestringStructure.get_plaquette`, this object
        specifies which matrix indices should be computed and which gate strings
        they correspond to.

    gateset : GateSet
        The gate set used to specify the probabilities

    spamlabels : list of strings
        The spam labels to extract probabilities for, e.g. ['plus']

    probs_precomp_dict : dict, optional
        A dictionary of precomputed probabilities.  Keys are gate strings
        and values are prob-dictionaries (as returned from GateSet.probs)
        corresponding to each gate string.

    Returns
    -------
    numpy array of shape ( len(spamlabels), len(effectStrs), len(prepStrs) )
        probability values corresponding to spamLabel and gate sequences
        where gateString is sandwiched between the each prep-fiducial, 
        effect-fiducial pair.
    """
    ret = _np.nan * _np.ones( (len(spamlabels),gsplaq.rows,gsplaq.cols), 'd')
    if probs_precomp_dict is None:
        if gateset is not None:
            for i,j,gstr in gsplaq:
                probs = gateset.probs(gstr)
                ret[:,i,j] = [probs[sl] for sl in spamlabels]
    else:
        for i,j,gstr in gsplaq:
            probs = probs_precomp_dict[gstr]
            ret[:,i,j] = [probs[sl] for sl in spamlabels]
    return ret

@smart_cached
def chi2_matrix(gsplaq, dataset, gateset, minProbClipForWeighting=1e-4,
                probs_precomp_dict=None):
    """
    Computes the chi^2 matrix for a base gatestring.

    Parameters
    ----------
    gsplaq : GatestringPlaquette
        Obtained via :method:`GatestringStructure.get_plaquette`, this object
        specifies which matrix indices should be computed and which gate strings
        they correspond to.

    dataset : DataSet
        The data used to specify frequencies and counts

    gateset : GateSet
        The gate set used to specify the probabilities and SPAM labels

    minProbClipForWeighting : float, optional
        defines the clipping interval for the statistical weight (see chi2fn).

    probs_precomp_dict : dict, optional
        A dictionary of precomputed probabilities.  Keys are gate strings
        and values are prob-dictionaries (as returned from GateSet.probs)
        corresponding to each gate string.

    Returns
    -------
    numpy array of shape ( len(effectStrs), len(prepStrs) )
        chi^2 values corresponding to gate sequences where
        gateString is sandwiched between the each prep-fiducial,
        effect-fiducial pair.
    """
    gsplaq_ds = gsplaq.expand_aliases(dataset)
    spamlabels = gateset.get_spam_labels()
    cntMxs  = total_count_matrix(gsplaq_ds, dataset)[None,:,:]
    probMxs = probability_matrices(gsplaq, gateset, spamlabels,
                                    probs_precomp_dict)
    freqMxs = frequency_matrices(gsplaq_ds, dataset, spamlabels)
    chiSqMxs= _tools.chi2fn( cntMxs, probMxs, freqMxs,
                                     minProbClipForWeighting)
    return chiSqMxs.sum(axis=0) # sum over spam labels


@smart_cached
def logl_matrix(gsplaq, dataset, gateset, minProbClip=1e-6,
                probs_precomp_dict=None):
    """
    Computes the log-likelihood matrix of 2*( log(L)_upperbound - log(L) )
    values for a base gatestring.

    Parameters
    ----------
    gsplaq : GatestringPlaquette
        Obtained via :method:`GatestringStructure.get_plaquette`, this object
        specifies which matrix indices should be computed and which gate strings
        they correspond to.

    dataset : DataSet
        The data used to specify frequencies and counts

    gateset : GateSet
        The gate set used to specify the probabilities and SPAM labels

    minProbClip : float, optional
        defines the minimum probability "patch-point" of the log-likelihood function.

    probs_precomp_dict : dict, optional
        A dictionary of precomputed probabilities.  Keys are gate strings
        and values are prob-dictionaries (as returned from GateSet.probs)
        corresponding to each gate string.


    Returns
    -------
    numpy array of shape ( len(effectStrs), len(prepStrs) )
        logl values corresponding to gate sequences where
        gateString is sandwiched between the each prep-fiducial,
        effect-fiducial pair.
    """
    gsplaq_ds = gsplaq.expand_aliases(dataset)
    
    spamlabels = gateset.get_spam_labels()
    cntMxs  = total_count_matrix(   gsplaq_ds, dataset)[None,:,:]
    probMxs = probability_matrices( gsplaq, gateset, spamlabels,
                                    probs_precomp_dict)
    freqMxs = frequency_matrices(   gsplaq_ds, dataset, spamlabels)
    logLMxs = _tools.two_delta_loglfn( cntMxs, probMxs, freqMxs, minProbClip)
    return logLMxs.sum(axis=0) # sum over spam labels



def small_eigval_err_rate(sigma, dataset, directGSTgatesets):
    """
    Compute per-gate error rate.

    The per-gate error rate, extrapolated from the smallest eigvalue
    of the Direct GST estimate of the given gate string sigma.

    Parameters
    ----------
    sigma : GateString or tuple of gate labels
        The gate sequence that is used to estimate the error rate

    dataset : DataSet
        The dataset used obtain gate string frequencies

    directGSTgatesets : dictionary of GateSets
        A dictionary with keys = gate strings and
        values = GateSets.

    Returns
    -------
    float
        the approximate per-gate error rate.
    """
    if sigma is None: return _np.nan # in plot processing, "None" gatestrings = no plot output = nan values
    gs_direct = directGSTgatesets[sigma]
    minEigval = min(abs(_np.linalg.eigvals( gs_direct.gates["GsigmaLbl"] )))
    return 1.0 - minEigval**(1.0/max(len(sigma),1)) # (approximate) per-gate error rate; max averts divide by zero error



def _eformat(f, prec):
    """
    Formatting routine for writing compact representations of
    numbers in plot boxes
    """
    if prec == 'compact' or prec == 'compacthp':
        if f < 0:
            return "-" + _eformat(-f,prec)

        if prec == 'compacthp':
            if f < 0.005: #can't fit in 2 digits; would just be .00, so just print "0"
                return "0"
            if f < 1:
                z = "%.2f" % f # print first two decimal places
                if z.startswith("0."): return z[1:]  # fails for '1.00'; then thunk down to next f<10 case
            if f < 10:
                return "%.1f" % f # print whole number and tenths

        if f < 100:
            return "%.0f" % f # print nearest whole number if only 1 or 2 digits

        #if f >= 100, minimal scientific notation, such as "4e7", not "4e+07"
        s = "%.0e" % f
        try:
            mantissa, exp = s.split('e')
            exp = int(exp)
            if exp >= 100: return "B" #if number is too big to print
            if exp >= 10: return "*%d" % exp
            return "%se%d" % (mantissa, exp)
        except:
            return str(s)[0:3]

    elif type(prec) == int:
        if prec >= 0:
            return "%.*f" % (prec,f)
        else:
            return "%.*g" % (-prec,f)
    else:
        return "%g" % f #fallback to general format

#OLD
#
#def _computeGateStringMaps(gss, dataset):
##    xvals, yvals, xyGateStringDict
##    strs, fidpair_filters, gatestring_filters,
##                           gateLabelAliases
#    """ 
#    Return a dictionary of all the gatestring maps,
#    indexed by base string. 
#    """
#    return { gss.gsDict[(x,y)] :
#                 get_gatestring_map(gss.gsDict[(x,y)], dataset, (gss.prepStrs, gss.effectStrs),
#                                    gss.get_fidpair_filter(x,y), gss.get_gatestring_filter(x,y),
#                                    gss.aliases)
#             for x in gss.used_xvals for y in gss.used_yvals }


def _num_non_nan(array):
    ixs = _np.where(_np.isnan(_np.array(array).flatten()) == False)[0]
    return int(len(ixs))


def _all_same(items):
    return all(x == items[0] for x in items)


def _compute_num_boxes_dof(subMxs, used_xvals, used_yvals, sumUp):
    """
    A helper function to compute the number of boxes, and corresponding
    number of degrees of freedom, for the GST chi2/logl boxplots.

    """
    if sumUp:
        s = _np.shape(subMxs)
        # Reshape the subMxs into a "flattened" form (as opposed to a
        # two-dimensional one)
        reshape_subMxs = _np.array(_np.reshape(subMxs, (s[0] * s[1], s[2], s[3])))

        #Get all the boxes where the entries are not all NaN
        non_all_NaN = reshape_subMxs[_np.where(_np.array([_np.isnan(k).all() for k in reshape_subMxs]) == False)]
        s = _np.shape(non_all_NaN)
        dof_each_box = [_num_non_nan(k) for k in non_all_NaN]

        # Don't assert this anymore -- just use average below
        if not _all_same(dof_each_box):
            _warnings.warn('Number of degrees of freedom different for different boxes!')

        # The number of boxes is equal to the number of rows in non_all_NaN
        n_boxes = s[0]

        if n_boxes > 0:
            # Each box is a chi2_(sum) random variable
            # dof_per_box = dof_each_box[0] #OLD
            dof_per_box = _np.average(dof_each_box)
        else:
            dof_per_box = None #unknown, since there are no boxes
    else:
        # Each box is a chi2_1 random variable
        dof_per_box = 1

        # Gets all the non-NaN boxes, flattens the resulting
        # array, and does the sum.
        n_boxes = _np.sum(~_np.isnan(subMxs).flatten())

    return n_boxes, dof_per_box

    

def _computeProbabilities(gss, gateset, dataset):
    """ 
    Returns a dictionary of probabilities for each gate sequence in
    GatestringStructure `gss`.
    """
    gatestringList = gss.allstrs

    #compute probabilities
    spamLabels = dataset.get_spam_labels()
    evt = gateset.bulk_evaltree(gatestringList)
    bulk_probs = gateset.bulk_probs(evt) # LATER use comm?
    probs_dict = \
        { gatestringList[i]: {sl: bulk_probs[sl][i] for sl in spamLabels}
          for i in range(len(gatestringList)) }
    return probs_dict

    
@smart_cached
def _computeSubMxs(gss, subMxCreationFn, sumUp):
    subMxs = [ [ subMxCreationFn(gss.get_plaquette(x,y),x,y)
                 for x in gss.used_xvals() ] for y in gss.used_yvals()]
    #Note: subMxs[y-index][x-index] is proper usage
    return subMxs


@smart_cached
def direct_chi2_matrix(gsplaq, gss, dataset, directGateset,
                       minProbClipForWeighting=1e-4):
    """
    Computes the Direct-X chi^2 matrix for a base gatestring sigma.

    Similar to chi2_matrix, except the probabilities used to compute
    chi^2 values come from using the "composite gate" of directGatesets[sigma],
    a GateSet assumed to contain some estimate of sigma stored under the
    gate label "GsigmaLbl".

    Parameters
    ----------
    gsplaq : GatestringPlaquette
        Obtained via :method:`GatestringStructure.get_plaquette`, this object
        specifies which matrix indices should be computed and which gate strings
        (for accessing the dataset) they correspond to.  

    gss : GatestringStructure
        The gate string structure object containing `gsplaq`.  The structure is
        neede to create a special plaquette for computing probabilities from the
        direct gateset containing a "GsigmaLbl" gate.

    dataset : DataSet
        The data used to specify frequencies and counts

    directGateset : GateSet
        GateSet which contains an estimate of sigma stored
        under the gate label "GsigmaLbl".

    minProbClipForWeighting : float, optional
        defines the clipping interval for the statistical weight (see chi2fn).


    Returns
    -------
    numpy array of shape ( len(effectStrs), len(prepStrs) )
        Direct-X chi^2 values corresponding to gate sequences where
        gateString is sandwiched between the each (effectStr,prepStr) pair.
    """
    spamlabels = dataset.get_spam_labels()
    plaq_ds = gsplaq.expand_aliases(dataset)
    plaq_pr = gss.create_plaquette( _objs.GateString( ("GsigmaLbl",) ) )
    
    cntMxs = total_count_matrix(plaq_ds, dataset)[None,:,:]
    probMxs = probability_matrices( plaq_pr, directGateset, spamlabels ) # no probs_precomp_dict
    freqMxs = frequency_matrices( plaq_ds, dataset, spamlabels)
    chiSqMxs= _tools.chi2fn( cntMxs, probMxs, freqMxs,
                                     minProbClipForWeighting)
    return chiSqMxs.sum(axis=0) # sum over spam labels



@smart_cached
def direct_logl_matrix(gsplaq, gss, dataset, directGateset,
                       minProbClip=1e-6):
    """
    Computes the Direct-X log-likelihood matrix, containing the values
     of 2*( log(L)_upperbound - log(L) ) for a base gatestring sigma.

    Similar to logl_matrix, except the probabilities used to compute
    LogL values come from using the "composite gate" of directGatesets[sigma],
    a GateSet assumed to contain some estimate of sigma stored under the
    gate label "GsigmaLbl".

    Parameters
    ----------
    gsplaq : GatestringPlaquette
        Obtained via :method:`GatestringStructure.get_plaquette`, this object
        specifies which matrix indices should be computed and which gate strings
        (for accessing the dataset) they correspond to.  

    gss : GatestringStructure
        The gate string structure object containing `gsplaq`.  The structure is
        neede to create a special plaquette for computing probabilities from the
        direct gateset containing a "GsigmaLbl" gate.

    dataset : DataSet
        The data used to specify frequencies and counts

    directGateset : GateSet
        GateSet which contains an estimate of sigma stored
        under the gate label "GsigmaLbl".

    minProbClip : float, optional
        defines the minimum probability clipping.

    Returns
    -------
    numpy array of shape ( len(effectStrs), len(prepStrs) )
        Direct-X logL values corresponding to gate sequences where
        gateString is sandwiched between the each (effectStr,prepStr) pair.
    """
    spamlabels = dataset.get_spam_labels()
    plaq_ds = gsplaq.expand_aliases(dataset)
    plaq_pr = gss.create_plaquette( _objs.GateString( ("GsigmaLbl",) ) )

    cntMxs = total_count_matrix(plaq_ds, dataset)[None,:,:]
    probMxs = probability_matrices( plaq_pr, directGateset, spamlabels ) # no probs_precomp_dict
    freqMxs = frequency_matrices( plaq_ds, dataset, spamlabels)
    logLMxs = _tools.two_delta_loglfn( cntMxs, probMxs, freqMxs, minProbClip)
    return logLMxs.sum(axis=0) # sum over spam labels



@smart_cached
def dscompare_llr_matrices(gsplaq, dscomparator):
    """
    Computes matrix of 2*log-likelihood-ratios comparing the 
    datasets of `dscomparator`.

    Parameters
    ----------
    gsplaq : GatestringPlaquette
        Obtained via :method:`GatestringStructure.get_plaquette`, this object
        specifies which matrix indices should be computed and which gate strings
        they correspond to.

    dscomparator : DataComparator
        The object specifying that data to be compared.

    Returns
    -------
    numpy array of shape ( len(effectStrs), len(prepStrs) )
        log-likelihood-ratio values corresponding to the gate sequences
        where a base gateString is sandwiched between the each prep-fiducial and
        effect-fiducial pair.
    """
    llrVals_and_strings_dict = dict(dscomparator.llrVals_and_strings)
    ret = _np.nan * _np.ones( (gsplaq.rows,gsplaq.cols), 'd')
    for i,j,gstr in gsplaq:
        ret[i,j] = llrVals_and_strings_dict[gstr]
    return ret
